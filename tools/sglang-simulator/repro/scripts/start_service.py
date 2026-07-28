#!/usr/bin/env python3
import argparse
import os

from common import build_server_args, configure_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-args", required=True)
    parser.add_argument("--hisim-config", required=True)
    parser.add_argument("--mode", choices=["OFFLINE", "BLOCKING"], default="OFFLINE")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--page-size", type=int)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    configure_environment(
        args.hisim_config, args.output_dir, args.mode, args.device
    )

    from sglang_simulator.simulation.sglang.hook_bootstrap import (
        install_simulator_hooks,
        run_simulator_detokenizer_process,
        run_simulator_scheduler_process,
    )

    install_simulator_hooks()
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.utils import kill_process_tree

    server_args = build_server_args(args.server_args, args.device, args.page_size)
    try:
        launch_server(
            server_args,
            run_scheduler_process_func=run_simulator_scheduler_process,
            run_detokenizer_process_func=run_simulator_detokenizer_process,
        )
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
