import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def profile_inference_time(
        session: ort.InferenceSession,
        n_runs: int,
        n_warmup_runs: int = 10,
) -> list[float]:
    sess_inputs = session.get_inputs()[0]
    input_name = sess_inputs.name
    dummy_inputs = np.random.randn(*sess_inputs.shape).astype(np.float32)
    
    for _ in range(n_warmup_runs):
        session.run(None, {input_name: dummy_inputs})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: dummy_inputs})
        times.append((time.perf_counter() - t0))

    return times


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
