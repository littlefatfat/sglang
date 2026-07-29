import json
import random
from pathlib import Path

import numpy as np
from sglang.benchmark.datasets.common import DatasetRow
from sglang.benchmark.datasets.random import sample_random_requests
from sglang.benchmark.datasets.sharegpt import sample_sharegpt_requests
from sglang_simulator.dataset import GenericRequest, SimpleDataset
from transformers import AutoTokenizer


def load_hisim_trace_rows(
    dataset_path: str | Path,
    *,
    num_requests: int | None = None,
    timestamp_scale: float = 1.0,
) -> list[DatasetRow]:
    """Load a HiSim JSONL trace with timestamps normalized to relative seconds."""
    if timestamp_scale <= 0:
        raise ValueError("timestamp_scale must be greater than zero")

    path = Path(dataset_path)
    trace_rows = []
    with path.open(encoding="utf-8") as trace_file:
        for line_no, line in enumerate(trace_file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            timestamp = row.get("created_time", row.get("timestamp"))
            if timestamp is None:
                raise ValueError(
                    f"{path}:{line_no}: missing created_time/timestamp"
                )

            input_ids = row.get("input_ids")
            if not isinstance(input_ids, list):
                raise ValueError(f"{path}:{line_no}: input_ids must be a list")
            input_length = row.get("input_length", len(input_ids))
            if input_length != len(input_ids):
                raise ValueError(
                    f"{path}:{line_no}: input_length != len(input_ids)"
                )

            output_length = row.get("output_length")
            if not isinstance(output_length, int) or output_length <= 0:
                raise ValueError(
                    f"{path}:{line_no}: output_length must be a positive integer"
                )
            trace_rows.append(
                (float(timestamp) / timestamp_scale, input_ids, output_length)
            )

    if not trace_rows:
        raise ValueError(f"empty trace: {path}")

    trace_rows.sort(key=lambda item: item[0])
    if num_requests is not None:
        trace_rows = trace_rows[:num_requests]
    if not trace_rows:
        raise ValueError("num_requests must be greater than zero")
    trace_start = trace_rows[0][0]
    return [
        DatasetRow(
            prompt=input_ids,
            prompt_len=len(input_ids),
            output_len=output_length,
            timestamp=timestamp - trace_start,
        )
        for timestamp, input_ids, output_length in trace_rows
    ]


def _to_simulator_dataset(
    rows: list[DatasetRow],
    *,
    use_timestamps: bool,
) -> SimpleDataset:
    return SimpleDataset(
        reqs=[
            GenericRequest(
                prompt=row.prompt if isinstance(row.prompt, str) else None,
                token_ids=row.prompt if isinstance(row.prompt, list) else None,
                input_length=row.prompt_len,
                output_length=row.output_len,
                custom_params=(
                    {"created_time": row.timestamp}
                    if use_timestamps and row.timestamp is not None
                    else {}
                ),
            )
            for row in rows
        ]
    )


def load_inprocess_workload(
    *,
    name: str,
    model_path: str,
    dataset_path: str | None,
    num_prompts: int,
    input_len: int,
    output_len: int,
    timestamp_scale: float,
    seed: int = 42,
) -> SimpleDataset:
    """Use SGLang's benchmark samplers and adapt their rows for HiSim."""
    random.seed(seed)
    np.random.seed(seed)

    if name == "trace":
        if not dataset_path:
            raise ValueError("--dataset is required for trace")
        rows = load_hisim_trace_rows(
            dataset_path,
            num_requests=num_prompts,
            timestamp_scale=timestamp_scale,
        )
        return _to_simulator_dataset(rows, use_timestamps=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if name == "sharegpt":
        if not dataset_path:
            raise ValueError("--dataset is required for sharegpt")
        rows = sample_sharegpt_requests(
            dataset_path=str(Path(dataset_path)),
            num_requests=num_prompts,
            tokenizer=tokenizer,
        )
    elif name == "random":
        rows = sample_random_requests(
            input_len=input_len,
            output_len=output_len,
            num_prompts=num_prompts,
            range_ratio=1.0,
            tokenizer=tokenizer,
            dataset_path="",
            random_sample=False,
            return_text=False,
        )
    else:
        raise ValueError(f"unknown workload: {name}")

    return _to_simulator_dataset(rows, use_timestamps=False)
