# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .text_spotter import TransformerPureDetector
from .video_text_spotter import TransformerPureVideoDetector

_EXCLUDE = {"torch", "ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]
