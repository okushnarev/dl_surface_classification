from pathlib import Path
import torch.nn as nn
import torch


def export_model_to_onnx(model: nn.Module, save_path: Path, input_shape: tuple[int, ...]) -> None:
    dummy_input = torch.zeros(input_shape)
    torch.onnx.export(
        model,
        (dummy_input,),
        save_path,
        opset_version=17,
        input_names=['input'],
        output_names=['output'],
    )
