from pathlib import Path

import onnx_tool


def profile_mac_and_params_count(onnx_path: Path) -> tuple[float, int]:
    model = onnx_tool.Model(onnx_path)
    model.graph.shape_infer()
    model.graph.profile()
    total_macs = model.graph.macs[0]
    total_params = model.graph.params
    return total_macs, total_params
