"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import logging
import os
from logging import Logger
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from enums import FeatName
from tqdm import tqdm


class BaseDataset(torch.utils.data.Dataset):
    """Parent dataset class for loading in and processing data for MoFo framework"""

    def __init__(
        self,
        split: str,
        data_dir: List[str],
        logger: Logger | None = None,
        seq_len: int = 600,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        features_to_load: Optional[List[str]] = None,
        load_multiperson: bool = False,
    ):
        """
        Args:
            split: train/val/test
            data_dir: directory to load the data from
            logger: logger to log the data
            seq_len: sequence length to load
            train_ratio: ratio of full dataset to use for train partition
            val_ratio: ratio of full dataset to use for val partition
            features_to_load: list of features to load (
                e.g. [FeatName.BODY.v, FeatName.AUDIO_SEPARATED.v, etc.])
            load_multiperson: whether to load multiperson data or not

        DATASET FORMAT

        Datasets are expected to come in to following folder structure, assuming
        data_dir as the root folder.
        level 1: capture_name--sequence_name--frame_start--frame_end
        level 2: subject_id
        level 3: feature_name

        Further, a dataset.json file exist in the root folder.
        The file has the following format:
        {
            subject_id_1: {
                sequence_name_1: {
                    "length": num_frames_in_sequence.
                    "id": unique_sequence_id.
                    "text": path_to_text_annotation, or higher level text annotation itself, if either exists.
                    "multiperson": list of other subject ids in the same scene, if interacting.
                    "audio": path to audio file if exists.
                },
                sequence_name_2: {
                    ...
                }
            },
            subject_id_2: {
                ...
            },
            ...
        }

        The subject_id is required to be 6 character long string.
        """
        assert features_to_load is not None, "features_to_load must be specified"
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seq_len = seq_len
        self.features_to_load = features_to_load
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.load_multiperson = load_multiperson

        #### load dataset info; load all of the info jsons containing the lengths of each sequences
        # data_dir is a list of paths, so here we are combining the dataset.json if there are multiple
        self.data_dir = []
        dataset_info = {}
        for data_dir_path in data_dir:
            self.data_dir.append(data_dir_path)
            curr_dataset_info = self._configure_dataset(data_dir_path)
            for subject_id, sequences in curr_dataset_info.items():
                if subject_id not in dataset_info:
                    dataset_info[subject_id] = {}
                for k in sequences:
                    sequences[k]["data_dir"] = data_dir_path
                dataset_info[subject_id].update(sequences)

        #### create train, val, or test partition
        self.dataset_info = self._split_dataset(dataset_info)

        #### create a mapping from global (dataset-level) frame index into sequences
        # idx2seq is a dictionary of {global_frame_index: (subject_id, sequence_name)}
        # idx_per_seq is a list of global_frame_index for each sequence
        # num_valid_segments is the total number of valid segments in the dataset
        seqs = [
            (subject_id, sequence_name)
            for subject_id in self.dataset_info.keys()
            for sequence_name in self.dataset_info[subject_id].keys()
        ]
        idx = 0
        self.idx2seq = {}
        self.idx_per_seq = []
        for seq in seqs:
            self.idx2seq[idx] = seq
            self.idx_per_seq.append(idx)
            idx += self.dataset_info[seq[0]][seq[1]]["length"] - self.seq_len + 1
        self.num_valid_segments = idx

        #### preload the dataset into memory
        self._preload_dataset()

    def _configure_dataset(self, data_dir: str):
        """load the dataset info json file
        Args:
            data_dir: directory to load the dataset info from
        """
        dataset_info_path = f"{data_dir}/dataset.json"
        with open(dataset_info_path, "r") as f:
            dataset_info = json.load(f)
        return dataset_info

    def _split_dataset(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """define the train/val/test split along actors/subjects
        Args:
            dataset_info: dataset info dictionary (see DATASET FORMAT above)
        """
        split_list = list(dataset_info.keys())
        train_partition = int(len(split_list) * self.train_ratio)
        val_partition = int(len(split_list) * self.val_ratio)
        test_partition = len(split_list) - train_partition - val_partition

        if self.split == "val":
            if val_partition == 0:
                raise Exception(
                    f"[base_dataloader.py] val partition can't be built: {self.val_ratio} of {len(split_list)} is 0."
                )
            split_dataset = {k: dataset_info[k] for k in split_list[train_partition : train_partition + val_partition]}
        elif self.split == "test":
            if test_partition == 0:
                test_ratio = 1.0 - self.train_ratio - self.val_ratio
                raise Exception(
                    f"[base_dataloader.py] test partition can't be built: {test_ratio} of {len(split_list)} is 0."
                )
            split_dataset = {k: dataset_info[k] for k in split_list[train_partition + val_partition :]}
        else:  # train
            if train_partition == 0:
                raise Exception(
                    f"[base_dataloader.py] train partition can't be built: {self.train_ratio} of {len(split_list)} is 0."
                )
            split_dataset = {k: dataset_info[k] for k in split_list[:train_partition]}

        # logging
        n_seqs = sum([len(v) for v in split_dataset.values()])
        self.logger.info(f"[base_dataloader.py] {self.split} dataset {len(split_dataset)} subjects, {n_seqs} sequences")

        return split_dataset

    def _get_chunk_data(
        self, subject_id: str, sequence_name: str, start_frame: int | None = None
    ) -> Dict[str, torch.Tensor]:
        """load a single sequence from the dataset
        Args:
            subject_id: subject id of the person to load
            sequence_name: name of the sequence to load
            start_frame: start frame index of the sequence to load
            end_frame: end frame index of the sequence to load
        """
        data = {}

        # logic to set the start and end frame according to settings
        start_frame = start_frame if start_frame is not None else 0
        end_frame = start_frame + self.seq_len

        # metadata
        data[FeatName.SUBJECT_ID.value] = subject_id
        data[FeatName.SEQUENCE_ID.value] = sequence_name
        data[FeatName.START_FRAME.value] = torch.tensor(start_frame, dtype=torch.long)
        data[FeatName.LENGTH.value] = torch.tensor(end_frame - start_frame, dtype=torch.long)

        # features
        for feat_name in self.features_to_load:
            if subject_id not in self.dataset:
                self.logger.warning(f"[dataloader.py] subject {subject_id} not found in dataset")
                return data
            _data = np.copy(self.dataset[subject_id][sequence_name][feat_name])
            if feat_name == FeatName.AUDIO_SEPARATED.value or feat_name == FeatName.AUDIO_RAW.value:
                if _data is None or _data.ndim == 0:
                    self.logger.warning("[dataloader.py] audio not found for sequence; setting to zeros")
                    data[feat_name] = torch.zeros(1, data[FeatName.LENGTH.value] * 1600, dtype=torch.float32)
                else:
                    # load the audio features which are 48kHz
                    _data = _data[:, start_frame * 1600 : end_frame * 1600]
                    data[feat_name] = torch.from_numpy(np.copy(_data))
            elif feat_name == FeatName.TEXT.value:
                # sometimes the text annotations are not available just store an empty dict
                # for instance, we don't annotate the first 10 seconds of the first sequence
                # and there are cases where we don't annotate the end of the sequence
                # NOTE: for the free form text, you will need to pad them such that they are the same length text
                _data = _data.item()
                if _data is None:
                    self.logger.warning("[dataloader.py] text not found for sequence")
                    _data = ""
                elif "overview" in _data:
                    # only broad annotations are given (not per subsection)
                    data[feat_name] = _data["overview"]
                else:
                    # text annotations are stored in seconds at 10 second increments
                    seq_frames = sequence_name.split("--")[-1]
                    seq_start_frame = int(seq_frames.split("-")[0])
                    text_start_frame = int((start_frame / 30) // 10 * 10) * 30
                    text_frame_index = str(text_start_frame + seq_start_frame)
                    offset_in_seconds = (text_start_frame - start_frame) / 30
                    anno_to_select = FeatName.ANNO_MOVEMENT.value
                    if not text_frame_index in _data or anno_to_select not in _data[text_frame_index]:
                        self.logger.warning(f"[dataloader.py] text '{anno_to_select}' not found; change hardcoding")
                        _data = ""
                    else:
                        random_idx = np.random.randint(0, len(_data[text_frame_index][anno_to_select]))
                        _data = _data[text_frame_index][anno_to_select][random_idx]
                        data["text_offset_in_seconds"] = offset_in_seconds
                    data[feat_name] = _data
            elif feat_name == FeatName.TEXT_HOLISTIC.value:
                _data = _data.item()
                if _data is None:
                    self.logger.warning("[dataloader.py] text not found for sequence")
                    _data = ""
                anno_to_select = FeatName.ANNO_SCENE_MOOD.value
                if anno_to_select not in _data:
                    self.logger.warning(f"[dataloader.py] text '{anno_to_select}' not found; change hardcoding")
                    _data = ""
                else:
                    random_idx = np.random.randint(0, len(_data[anno_to_select]))
                    _data = _data[anno_to_select][random_idx]
                data[feat_name] = _data
            else:
                # load all the other smplx body features
                _data = _data[start_frame:end_frame]
                assert _data.shape[0] == self.seq_len, "data shape is not seq len"
                data[feat_name] = torch.from_numpy(np.copy(_data))
        return data

    def _preload_dataset(self, quiet: bool = False) -> None:
        """Preload the dataset to avoid loading the same data multiple times"""
        dataset = {}
        seqs = [
            (subject_id, sequence_name)
            for subject_id in self.dataset_info.keys()
            for sequence_name in self.dataset_info[subject_id].keys()
        ]

        #### load the data by iterating through the sequences inside of the dataset info json
        for subject_id, sequence_name in tqdm(seqs, desc=f"Preloading {self.split} dataset", disable=quiet):
            if subject_id not in dataset:
                dataset[subject_id] = {}
            base_sequence_name = os.path.splitext(sequence_name)[0]
            data_dir = self.dataset_info[subject_id][sequence_name]["data_dir"]
            seq_dict = dataset[subject_id].get(sequence_name, {})
            for feat_name in self.features_to_load:
                if feat_name == FeatName.AUDIO_SEPARATED.value:
                    audio_anno = self.dataset_info[subject_id][sequence_name]["audio"]
                    if audio_anno is None:
                        seq_dict[feat_name] = None
                    else:
                        seq_dict[feat_name] = torchaudio.load(audio_anno)[0].to(torch.float32)
                elif feat_name == FeatName.AUDIO_RAW.value:
                    # NOTE: assumes a certain data format to calculate the audio path for mono
                    # the default file structure upon untaring the dataset
                    separated_audio_anno = self.dataset_info[subject_id][sequence_name]["audio"]
                    if separated_audio_anno is None:
                        seq_dict[feat_name] = None
                    else:
                        audio_anno = os.path.dirname(separated_audio_anno.split("audio_separated")[0][:-1])
                        audio_anno = os.path.join(audio_anno, "audio_raw")
                        audio_basename = os.path.basename(separated_audio_anno)
                        audio_basename = "--".join(os.path.splitext(audio_basename)[0].split("--")[-2:])
                        mono_audio_path = f"{audio_anno}/{audio_basename}.wav"
                        seq_dict[feat_name] = torchaudio.load(mono_audio_path)[0].to(torch.float32)
                elif feat_name == FeatName.TEXT.value:
                    text_anno = str(self.dataset_info[subject_id][sequence_name]["text"])
                    if text_anno is None:
                        seq_dict[feat_name] = {}
                    elif text_anno.endswith(".json"):
                        with open(text_anno, "r") as f:
                            seq_dict[feat_name] = json.load(f)
                    else:
                        seq_dict[feat_name] = {"overview": text_anno}
                elif feat_name == FeatName.TEXT_HOLISTIC.value:
                    # NOTE: assumes a certain data format to calculate the text path for holistic
                    # the default file structure upon untaring the dataset
                    text_anno = str(self.dataset_info[subject_id][sequence_name]["text"])
                    if text_anno is None:
                        seq_dict[feat_name] = {}
                    else:
                        text_anno_separated = os.path.dirname(text_anno.split("text_annotations")[0][:-1])
                        text_anno_separated = os.path.join(text_anno_separated, "text_annotations_holistic")
                        text_basename = os.path.basename(text_anno)
                        text_basename = "--".join(os.path.splitext(text_basename)[0].split("--")[-2:])
                        holistic_text_path = f"{text_anno_separated}/{text_basename}.json"
                        if os.path.exists(holistic_text_path):
                            with open(holistic_text_path, "r") as f:
                                seq_dict[feat_name] = json.load(f)
                        else:
                            seq_dict[feat_name] = {}
                else:
                    data_feature = np.load(
                        f"{data_dir}/{sequence_name}/{subject_id}/{feat_name}/{base_sequence_name}.npy"
                    )
                    seq_dict[feat_name] = data_feature.astype(np.float32)
            dataset[subject_id][sequence_name] = seq_dict
        self.dataset = dataset
        self.logger.info(
            f"[base_dataloader.py] preloaded dataset with {len(seqs)} sequences, {self._calc_hours()} hours"
        )

    def __len__(self) -> int:
        return self.num_valid_segments

    def _calc_hours(self, fps: int = 30) -> float:
        """Calculates the number of hours in the dataset"""
        num_frames = self.num_valid_segments + self.seq_len - 1
        return num_frames / fps / 3600.0

    def _idx2segment(self, idx: int) -> Tuple[str, str, int]:
        """Get the subject_id, sequence_name, and frame_offset for a given index"""
        closest_idx = np.searchsorted(list(self.idx2seq.keys()), idx, side="right") - 1
        closest_idx = list(self.idx2seq.keys())[closest_idx]
        subject_id, sequence_name = self.idx2seq[closest_idx]
        frame_offset = idx - closest_idx
        return subject_id, sequence_name, frame_offset

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset"""
        subject_id, sequence_name, frame_offset = self._idx2segment(idx)
        data = self._get_chunk_data(subject_id, sequence_name, start_frame=frame_offset)
        if self.load_multiperson:
            if self.dataset_info[subject_id][sequence_name] is None:
                self.logger.warning(f"[dataloader.py] multiperson data not found for {subject_id} {sequence_name}")
            else:
                other_ids = self.dataset_info[subject_id][sequence_name]["multiperson"]
                other_id_dict = {}
                if other_ids is not None:
                    for other_id in other_ids:
                        other_id_dict[other_id] = self._get_chunk_data(
                            other_id, sequence_name, start_frame=frame_offset
                        )
                data[FeatName.MULTIPERSON.value] = other_id_dict
        return data
