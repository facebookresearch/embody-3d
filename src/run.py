"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import os

import ffmpeg
import smplx
import torch
import torchaudio
from dataloader import BaseDataset
from enums import FeatName
from torch.utils.data import DataLoader
from torchvision.io import write_video
from visualize import PyrenderRenderer


def compute_lbs_from_batch(args, smplx_model, batch, smplx_keys):
    smplx_input = {}
    for key in batch.keys():
        if key in smplx_keys:
            smplx_input[key.replace("smplx_mesh_", "")] = batch[key][0].to(args.device)
    # compute the vertices and render the video
    result_verts = smplx_model(**smplx_input).vertices[None, ...]
    return result_verts

def load_and_visualize_data(args):
    ### set up the dataset
    smplx_keys = [
        FeatName.BODY.value,
        FeatName.ROT.value,
        FeatName.TRANS.value,
        FeatName.SHAPE.value,
        FeatName.LEFT_HAND.value,
        FeatName.RIGHT_HAND.value,
    ]
    features_to_load = smplx_keys + args.anno

    dataset = BaseDataset(
        split="train",
        data_dir=args.data_path,
        seq_len=args.seq_len,
        features_to_load=features_to_load,
        load_multiperson=args.load_multiperson,
    )

    ### set up the smplx function
    smplx_model = smplx.create(
        args.smplx_model_path,
        model_type="smplx",
        gender="neutral",
        flat_hand_mean=True,
        num_betas=300,
        num_expression_coeffs=100,
        use_pca=False,
        batch_size=args.seq_len,
    ).to(args.device)

    ### set up the renderer for smplx
    r = PyrenderRenderer(args.smplx_topology_path).to(args.device)
    r.faces = r.faces[:, [0, 2, 1]]

    ### set up to the dataloader
    dl_kwargs = {"num_workers": 0, "shuffle": True, "batch_size": 1}
    dataloader = DataLoader(dataset, **dl_kwargs)

    ### iterate through the dataloader
    for i, batch in enumerate(dataloader):
        print(f".....[batch {i}]:.....")
        print(f"subject_id: {batch['subject_id'][0]}")
        print(f"sequence_id: {batch['sequence_id'][0]}")
        print(f"loaded batch keys: {batch.keys()}")
        # rename the smplx keys so that it matches the smplx function
        verts_list = []
        print(f"rendering smplx mesh for {batch['subject_id'][0]}")
        verts_list.append(compute_lbs_from_batch(args, smplx_model, batch, smplx_keys))
        assert verts_list[0].shape == torch.Size([1, args.seq_len, 10475, 3])
        if FeatName.MULTIPERSON.value in batch.keys():
            other_ids = batch[FeatName.MULTIPERSON.value]
            for other_id in other_ids.keys():
                print(f"multiperson: rendering smplx mesh for {other_id}")
                other_batch = other_ids[other_id]
                verts_list.append(compute_lbs_from_batch(args, smplx_model, other_batch, smplx_keys))
        video_frames = r(verts_list)[0]  # T x 3 x H x W
        video_frames = video_frames.transpose(0, 2, 3, 1)  # T x H x W x 3
        save_path = (
            f"{args.output_dir}/{batch['subject_id'][0]}--{batch['sequence_id'][0]}-{batch['start_frame'][0]}.mp4"
        )
        write_video(save_path, video_frames, fps=30)
        if FeatName.TEXT.value in batch.keys():
            print(f"text anno: {batch[FeatName.TEXT.value][0]}")
            if "text_offset_in_seconds" in batch.keys():
                print(f"text anno offset in seconds: {batch['text_offset_in_seconds'][0]:02f}")
        if FeatName.TEXT_HOLISTIC.value in batch.keys():
            print(f"text anno holistic: {batch[FeatName.TEXT_HOLISTIC.value][0]}")
        if (
            FeatName.AUDIO_SEPARATED.value in batch.keys() and torch.any(batch[FeatName.AUDIO_SEPARATED.value][0] != 0)
        ) or (FeatName.AUDIO_RAW.value in batch.keys() and torch.any(batch[FeatName.AUDIO_RAW.value][0] != 0)):
            key = FeatName.AUDIO_RAW if FeatName.AUDIO_RAW.value in batch.keys() else FeatName.AUDIO_SEPARATED
            print(f"found {key.value}, if trying to load separated and mono, mono takes precedence")
            audio_path = f"{os.path.splitext(save_path)[0]}_audio.wav"
            video_audio_path = f"{os.path.splitext(save_path)[0]}_audio.mp4"
            torchaudio.save(audio_path, batch[key.value][0], 48_000)  # sr = 48_000 from the file
            video_input = ffmpeg.input(save_path)
            audio_input = ffmpeg.input(audio_path)
            ffmpeg_out = ffmpeg.output(video_input, audio_input, video_audio_path, vcodec="copy", acodec="aac")
            ffmpeg_out = ffmpeg_out.global_args("-hide_banner", "-loglevel", "error")
            result = ffmpeg_out.run(overwrite_output=True)
            print(f"ffmpeg result: {result}")
            print(f"saved video with audio {video_audio_path}\n")
        else:
            print(f"saved video without audio {save_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smplx_model_path", type=str, required=True, help="path to the smplx model")
    parser.add_argument("--smplx_topology_path", type=str, required=True, help="path to the smplx obj topology")
    parser.add_argument("--output_dir", type=str, required=True, help="output directory to save the rendered videos")
    parser.add_argument("--load_multiperson", action="store_true", help="whether to load the multiperson dataset")
    parser.add_argument("--data_path", nargs="+", type=str, required=True, help="A list of data directories to load")
    parser.add_argument("--anno", nargs="+", type=str, default=[], help="additional annotations to load, e.g. audio")
    parser.add_argument("--seq_len", type=int, default=600, help="max length of the sequences to load")
    parser.add_argument("--device", type=str, default="cuda", help="device to run the smplx model on")
    args = parser.parse_args()
    # example script to load and render a batch of data
    load_and_visualize_data(args)
