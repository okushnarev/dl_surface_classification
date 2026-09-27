import time
from pathlib import Path

import codegreen
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


def profile_energy_usage(session: ort.InferenceSession, n_runs: int = 100) -> float:
    profile_inference_time(session, 0, 10)
    task_name = 'forward_pass'
    with codegreen.Session('onnx_inference', save_to_file=False) as s:
        with s.task(task_name):
            profile_inference_time(session, n_runs, 0)
    energy_per_run = None
    for task in s.tasks:
        if task.name == task_name:
            energy_per_run = task.energy_j / n_runs
    return energy_per_run
