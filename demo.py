# https://github.com/openvinotoolkit/anomalib

"""Anomalib Gradio Script.

This script provide a gradio web interface
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import gradio as gr
import numpy as np

from anomalib.deploy import Inferencer
from anomalib.deploy.inferencers.torch_inferencer import TorchInferencer


def get_parser() -> ArgumentParser:
    """Get command line arguments.

    Example:

        Example for Torch Inference.
        >>> python tools/inference/gradio_inference.py  \
        ...     --weights ./results/padim/mvtec/bottle/weights/torch/model.pt

    Returns:
        ArgumentParser: Argument parser for gradio inference.
    """
    parser = ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights")
    parser.add_argument("--metadata", type=Path, required=False, help="Path to a JSON file containing the metadata.")
    parser.add_argument("--share", type=bool, required=False, default=False, help="Share Gradio `share_url`")

    return parser


def get_inferencer(weight_path: Path, metadata: Path | None = None) -> Inferencer:
    """Parse args and open inferencer.

    Args:
        weight_path (Path): Path to model weights.
        metadata (Path | None, optional): Metadata is required for OpenVINO models. Defaults to None.

    Raises:
        ValueError: If unsupported model weight is passed.

    Returns:
        Inferencer: Torch or OpenVINO inferencer.
    """

    # Get the inferencer. We use .ckpt extension for Torch models and (onnx, bin)
    # for the openvino models.
    inferencer = TorchInferencer(path=weight_path)

    return inferencer


def infer(image: np.ndarray, inferencer: Inferencer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inference function, return anomaly map, score, heat map, prediction mask ans visualisation.

    Args:
        image (np.ndarray): image to compute
        inferencer (Inferencer): model inferencer

    Returns:
        tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        heat_map, pred_mask, segmentation result.
    """
    # Perform inference for the given image.
    predictions = inferencer.predict(image=image)
    return (predictions.heat_map, predictions.pred_mask, predictions.segmentations)


if __name__ == "__main__":
    args = get_parser().parse_args()
    gradio_inferencer = get_inferencer(args.weights, args.metadata)

    interface = gr.Interface(
        fn=lambda image: infer(image, gradio_inferencer),
        inputs=[
            gr.Image(
                shape=None, image_mode="RGB", source="upload", tool="editor", type="numpy", label="Image"
            ),
        ],
        outputs=[
            gr.Image(type="numpy", label="Predicted Heat Map"),
            gr.Image(type="numpy", label="Predicted Mask"),
            gr.Image(type="numpy", label="Segmentation Result"),
        ],
        title="Anomalib",
        description="Anomalib Gradio",
    )

    interface.launch(share=args.share)