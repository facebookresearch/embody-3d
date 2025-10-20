"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from enum import Enum


class FeatName(Enum):
    """list of possible features for all datasets"""

    # anno available
    BODY = "smplx_mesh_body_pose"
    ROT = "smplx_mesh_global_orient"
    TRANS = "smplx_mesh_transl"
    SHAPE = "smplx_mesh_betas"
    LEFT_HAND = "smplx_mesh_left_hand_pose"
    RIGHT_HAND = "smplx_mesh_right_hand_pose"
    AUDIO_SEPARATED = "audio_separated"
    AUDIO_RAW = "audio_raw"  # mono audio
    TEXT = "text_annotations"
    MULTIPERSON = "other_ids"
    TEXT_HOLISTIC = "text_annotations_holistic"

    # text annotation keys
    ANNO_POSTURE_SELECT = "describe_person_posture_multiple_select"
    ANNO_POSTURE_FREEFORM = "describe_person_posture_free_form"
    ANNO_RHYTHM_FREEFORM = "describe_person_rhythm_of_motion_explained"
    ANNO_RHYTHM_SELECT = "describe_person_rhythm_of_motion"
    ANNO_MOOD_SELECT = "describe_person_mood_multiple_select"
    ANNO_MOOD_FREEFORM = "describe_person_mood_free_form"
    ANNO_MOVEMENT = "describe_person_movement"
    ANNO_ACTION = "describe_person_action"
    ANNO_ACTIVE_PASSIVE = "is_active_or_passive_person"
    ANNO_TOUCH_OBJECT = "is_person_touching_objects"
    ANNO_TOUCH_PERSON = "is_person_touching_people"

    # text annoation keys for holistic
    ANNO_SCENE_EXPLIANED = 'scene_explained'
    ANNO_SCENE_MOOD_SELECT = 'scene_mood'
    ANNO_SCENE_MOOD = 'scene_mood_explained'
    ANNO_SCENE_ATMOSPHERE_SELECT = 'scene_atmosphere'
    ANNO_SCENE_ATMOSPHERE = 'scene_atmosphere_explained'
    ANNO_SCENE_RELATIONSHIP = 'describe_relationship_between_people'

    # meta data
    SUBJECT_ID = "subject_id"
    SEQUENCE_ID = "sequence_id"
    START_FRAME = "start_frame"
    LENGTH = "length"
