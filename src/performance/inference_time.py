from pathlib import Path

import onnxruntime as ort


def setup_ort_session(
        model_path: Path,
        intra_op_num_threads: int = 4,
        inter_op_num_threads: int = 1,
        providers: list[str] = ['CPUExecutionProvider'],
        **session_kwargs
) -> ort.InferenceSession:
    sess_options = ort.SessionOptions()

    sess_options.intra_op_num_threads = intra_op_num_threads
    sess_options.inter_op_num_threads = inter_op_num_threads
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    for k, v in session_kwargs.items():
        setattr(sess_options, k, v)

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=providers,
    )

    return session
