# Embody 3D ✨

Official repo for [Embody 3D dataset](https://www.meta.com/emerging-tech/codec-avatars/embody-3d/).
Included in this rep:
* 📝 [Overview](#overview): information about the dataset
* ⏬ [Download Data](#download-data): scripts to obtain the full dataset or subsets of the dataset
* 🔖 [Dataset Description](#dataset-description): high level overview of how the dataset is formatted
* 💻 [Explore the Dataset](#explore-the-dataset): tutorial for running a basic dataloader with the dataset and rendering videos

![Annotation GIF](assets/loop_fade.gif)

## Overview
The Codec Avatars Lab at Meta introduces Embody 3D, a multimodal dataset of 500 individual hours
of 3D motion data from 439 participants collected in a multi-camera collection stage, amounting to
over 54 million frames of tracked 3D motion. The dataset features a wide range of single-person motion
data, including prompted motions, hand gestures, and locomotion; as well as multi-person behavioral
and conversational data like discussions, conversations in different emotional states, collaborative
activities, and co-living scenarios in an apartment-like space. We provide tracked human motion
including hand tracking and body shape, text annotations, and a separate audio track for each
participant.


The following table illustrates what each section includes in terms of hours and annotations.
Please refer to the [download](#download-data) section for how to retrieve the dataset.
### Annotation Table
|   Section  | Hours | Body Shape | Hands | Audio | Text | MultiPerson |
|--------------|-------|------------|------|---|---|---|
| Charades 🎲| 88.9 | ✔️ | ✔️ |  <span style="color: red;">&#10006;</span> | ✔️|  <span style="color: red;">&#10006;</span> |
| Hand Interactions 🙌 | 111.3 | ✔️ | ✔️ | <span style="color: red;">&#10006;</span> | <span style="color: red;">&#10006;</span> | <span style="color: red;">&#10006;</span> |
| Locomotion 🚶‍♀️ | 21.0 | ✔️ | ✔️ | <span style="color: red;">&#10006;</span> | (✔️) | <span style="color: red;">&#10006;</span> |
| Dyadic Conversations 🧑‍🤝‍🧑 | 59.4 | ✔️ | ✔️ | ✔️ | (✔️) | ✔️ |
| Multi-Person Conversations 👭🧍‍♂️| 125.2 | ✔️ | ✔️ |✔️ | <span style="color: red;">&#10006;</span> | ✔️ |
| Scenarios 🛠️ | 49.2 | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ |
| Day in the Life 🛌 |  46.4 | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ |

(✔️) indicates that there is only high level text information (e.g. emotion annotations)
In these cases, the text annotation is in the filename of the sequence.


## Download Data

To download the full dataset, you must first get access by filling out the [release form](https://www.meta.com/emerging-tech/codec-avatars/embody-3d)

>[!WARNING]
>If you do not fill out the release form, you will be blocked from downloading the files, and the download script will not work.

Once you have filled out the release form, you will get a list of 21 download links.
**Please copy the download links into a .txt file.**
For instance, if you copy the links to a file `download.txt`, you can download the complete data by running the following:
```
python src/download.py --src download.txt
```

If you want to download only certain features, you can set the flag `--feat`. For instance if you want to download only the text
```
python src/download.py --src download.txt --feat text
```

Similarly, if you want to donwload only certain categories, you an set the flag `--category`. For instance, to just download charades,
```
python src/download.py --src download.txt --category charades
```

>[!Tip]
>You can further do more complex selections by combining the `--feat` and `--category` flags

## Dataset Description
### File Structure: how the data is laid out...
If you are downloading the acting subset for instance, this will unpack into the following format:
```
datasets
|-- acting
    |-- c--20250108--1300--DXG448--SZM479--JON169--BWW760--pilot--MotionPrior--ACTING_Adult_Birthday_--103301-106600
        |-- videos
            |-- ACTING_Adult_Birthday_--103301-106600.mp4
        |-- text_annotations_holistic
            |-- ACTING_Adult_Birthday_--103301-106600.json
        |-- BWW760
            |-- missing
                |-- c--20250108--1300--DXG448--SZM479--JON169--BWW760--pilot--MotionPrior--ACTING_Adult_Birthday_--103301-106600.npy
            |-- separated_audio
                |-- c--20250108--1300--DXG448--SZM479--JON169--BWW760--pilot--MotionPrior--ACTING_Adult_Birthday_--103301-106600.wav
            |-- smplx_mesh_betas
                |-- c--20250108--1300--DXG448--SZM479--JON169--BWW760--pilot--MotionPrior--ACTING_Adult_Birthday_--103301-106600.npy
            |-- smplx_mesh_body_pose
            |-- smplx_mesh_global_orient
            |-- smplx_mesh_left_hand_pose
            |-- smplx_mesh_right_hand_pose
            |-- smplx_mesh_transl
            |-- text_annotations
                |-- c--20250108--1300--DXG448--SZM479--JON169--BWW760--pilot--MotionPrior--ACTING_Adult_Birthday_--103301-106600.json
        |-- DXG448
        |-- JON169
        |-- SZM479
    |-- ...
    |-- dataset.json
```

>[!Note]
>When downloading the 7 categories, **some are composed of more subsections**. Eg. multiperson is composed of "emotions", "location", "polyadic", and "icebreakers".
>The untar will automatically split into these subsections, and each will have it's own `dataset.json`.
>You can continue to [Expore the Dataset](#add-ons-important-flags-to-know-about) (see item 4) to see how to combine these sections.

### Feature Descriptions: how to interpret the folders/files...
These are the features that every capture directory has:
1. `videos/` A video where you can see the scene from a birds-eye-view camera.
2. `missing/` Binary indicator to show which smplx frames are corrupted (0 indicates do not use, 1 indicates good tracking)
3. `smplx_mesh_*/` All of the smplx features needed to render out the mesh
4. `dataset.json` An overview of every capture sequence and what assets it has. It is formatted in the following:
```
{
    id_name: {
        capture_name: {
            length: number of frames (30fps) in this sequence
            id: a unique number associated with this capture_name
            text: either the path to the text annotation .json if it exists, or the short-form text annotation itself. If None, text annotation is not available.
            multiperson: list of id_names of other participants involved in the capture. If None, multi-person dynamics is not a part of this capture.
            audio: path to sound separated audio for this participant id. If None, audio is not available.
        }
    }
}
```

These are the features that some capture directories have:
1. `text_annotations_holistic/` High level text that was manually annotated and describe the entire capture sequence (e.g. mood/theme/etc.) Only provided for acting and daylife.
2. `separated_audio/` Speaker separated .wav file for the given individual.
3. `text_annotations/` Text that was manually annotated and describe 10 second chunks at a time. More mid-level text descriptions of the motion. Only provided for acting and daylife. For annotations that were indicated with a (✔️) in the [annotation table](#annotation-table) above, the text annotation is derived from the sequence name itself. Eg. `c--20250508--1123--ZJW644--OTR353--pilot--MotionPrior3--LOCOMOTION_high_kicks--029552-030448.npy` corresponds to high_kicks.
For simplicity, we have included these segments in the `dataset.json`.
There are also a few segments in the `dataset.json` that will have a "class label" stored as an int as opposed to text descriptions.
You can see an example of how to load this in the [dataset section](#add-ons-important-flags-to-know-about) (see item 2).


## Explore the Dataset
### Run Script: how to load and render the data...
First, follow this [smplx repo](https://github.com/vchoutas/smplx) to install the library for smplx.
Make sure you also download the [smplx assets](https://smpl-x.is.tue.mpg.de/) which will give you the .npz files.

Once you have downloaded the data, we provide an example dataloader along with a small visualization script for the dataset to render out the meshes with audio.
```
python src/run.py \
    --smplx_model_path assets/smplx/smplx_models_lockedhead/ \
    --smplx_topology_path assets/smplx/smplx_mesh.obj \
    --data_path <path_to_dataset>/acting/ \
    --output_dir /tmp/
```
This will load the *acting* sequences and then save them to the `/tmp/` directory.
**Please replace the \<path to dataset\> with your data directory path.**
You can change the `--data_path` to get a different section loaded, and change the `--output_dir` accordingly.

>[!Tip]
>You can access the `datasets/*/dataset.json` in for each subsection from this github. This file stores all the paths to the annotations. If the annotation is `None` for a given sequence, that means the annotation does not exist.

By default, this will load only the assets required to render out the smplx.

### Add-ons: important flags to know about...
We provide additional functionality for the dataloader.
1. If you want to load more, you can add to the `--anno` flag.
    For instance, if you wanted to load text annotations and the audio annotations, add
    ```
    --anno text_annotations audio_separated
    ```
    to above the command. You can exclude one or the other by removing it from the list.
2. For portions where there are multiple people, you can also add the flag
   ```
   --load_multiperson
   ```
   to load the assets of the invidiuals interacting in the scene.
3. If you want to load all possible assets, you can run the following to load all possible assets.
   However, since not all annotations are available for every section,
    the run file will print a warning when the asset is not found.
    ```
    python src/run.py \
        --smplx_model_path assets/smplx/smplx_models_lockedhead/ \
        --smplx_topology_path assets/smplx/smplx_mesh.obj \
        --data_path <path_to_dataset>/acting/ \
        --output_dir /tmp/ \
        --anno text_annotations text_annotations_holistic audio_separated audio_raw \
        --load_multiperson
    ```
4. If you want to combine subsections for the dataloader, you can chain the paths from the argument `--data_path`. e.g.
    ```
    --data_path <path_to_dataset>/acting/ <path_to_dataset>/daylife/
    ```
5. You can also change the total length of the sequences by passing in the flag `--max_seq_length`, followed by the number of frames.
    The default is set to 600 frames (20 seconds).

>[!Tip]
> If you immediately get a No such file or directory, you are likely not passing in the correct annotation label or your file strucutre is wrong.
> make sure your data path points to one of the categories e.g. acting/ emotions/

>[!Note]
>Since not all annotations are available for every dataset, please review [annotation table](#annotation-table) to see if the provided annotations are suitable for your task.

## Citation

If you use this dataset, please consider citing this work via:

```
@techreport{mclean2025embody3d,
  title     = {Embody 3D: A Large-scale Multimodal Motion and Behavior Dataset},
  author    = {Claire McLean and Makenzie Meendering and Tristan Swartz and Orri Gabbay and Alexandra Olsen and Rachel Jacobs and Nicholas Rosen and Philippe de Bree and Tony Garcia and Gadsden Merrill and Jake Sandakly and Julia Buffalini and Neham Jain and Steven Krenn and Moneish Kumar and Dejan Markovic and Evonne Ng and Fabian Prada and Andrew Saba and Siwei Zhang and Vasu Agrawal and Tim Godisart and Alexander Richard and Michael Zollhoefer},
  institution = {arXiv},
  year      = {2025},
  type      = {Technical Report},
  note      = {arXiv preprint},
}
```

## License

This toolbox presented in this repository is licensed under the [LICENSE](https://github.com/facebookresearch/embody-3d/blob/main/LICENSE). Note that the dataset itself is licensed under the XRCIA license, see the dataset webpage.
