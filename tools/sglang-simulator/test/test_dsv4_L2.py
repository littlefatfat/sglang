import os
from random import randint

from sglang_simulator.dataset import DatasetArgs, get_dataset
from sglang_simulator.simulation.benchmark import BenchmarkConfig
from transformers import AutoTokenizer

os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = (
    os.path.dirname(__file__) + "/assets/config.json"
)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["SGLANG_ENABLE_UNIFIED_RADIX_TREE"] = "1"


from sglang_simulator.simulation.sglang.bench_runner import (
    SGLangBenchmarkRunner,
)


def test_benchmark_sglang():
    from sglang.srt.server_args import ServerArgs  # noqa

    model_path = "/nfs/models/sgl-project/DeepSeek-V4-Flash-FP8"
    # model_path = "/nfs/models/Qwen/Qwen3-8B"
    runner = SGLangBenchmarkRunner(
        server_args=ServerArgs(
            model_path=model_path,
            load_format="dummy",
            device="cpu",
            enable_hierarchical_cache=True,
            hicache_ratio=2.0,
            hicache_io_backend="direct",
            max_total_tokens=200000,
            page_size=256,
        )
    )

    min_id = 1000
    max_id = 10000

    input_length_1 = 2000
    input_ids_1 = [3001] * 1024 + [randint(min_id, max_id) for _ in range(input_length_1 - 1024)]
    simulation_params_1 = {
        "total_request": 1,  # include the warmup requests.
        "created_time": 0,
    }

    engine = runner.engine
    ret_1 = engine.generate(
        input_ids=input_ids_1,
        sampling_params={
            "ignore_eos": True,
            "max_new_tokens": 10,
            "custom_params": {
                # (tmp) Transfer simulation arguments to the scheduler through the custom_params in sampling_params
                "simulation": simulation_params_1
            },
        },
    )

    input_length_2 = 2500
    input_ids_2 = [3001] * 1280 + [randint(min_id, max_id) for _ in range(input_length_2 - 1280)]
    simulation_params_2 = {
        "total_request": 1,  # include the warmup requests.
        "created_time": 20,
    }
    ret_2 = engine.generate(
        input_ids=input_ids_2,
        sampling_params={
            "ignore_eos": True,
            "max_new_tokens": 10,
            "custom_params": {
                # (tmp) Transfer simulation arguments to the scheduler through the custom_params in sampling_params
                "simulation": simulation_params_2
            },
        },
    )

    runner.shutdown()


if __name__ == "__main__":
    test_benchmark_sglang()
