# SGLang Simulator baseline collection

This launcher runs a real SGLang server and records the request stream and every
device-synchronized `ScheduleBatch`. The resulting JSONL files are the inputs for
simulator accuracy comparisons, replay predictors, and the ML latency-predictor
trainer.

## Start an instrumented server

Install the simulator from the same monorepo checkout, choose a new output
directory, and pass the normal SGLang server arguments:

```bash
pip install -e tools/sglang-simulator

python3 -m sglang_simulator.baseline.launch_server \
  --collection-output-dir /tmp/sglang-baseline/case-001 \
  --model-path /path/to/model \
  --disable-overlap-schedule \
  --tp 4
```

`--disable-overlap-schedule` is required. The collector synchronizes the device
before and after `Scheduler.run_batch`; it is intended for measurement runs and
will reduce serving throughput.

The output directory must not contain files. This prevents a later run from
silently overwriting a previous baseline.

## Run traffic

Use the standard SGLang serving benchmark. `--profile` starts collection after
benchmark warmup and exports it after all measured requests complete:

```bash
python3 -m sglang.benchmark.serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 32 \
  --num-prompts 1000 \
  --request-rate 10 \
  --profile \
  --output-file /tmp/sglang-baseline/case-001/metrics.json
```

For a custom traffic driver, call `/start_profile` immediately before measured
traffic and `/stop_profile` after it completes. Start clears in-memory records;
stop writes the files and disables collection.

Use a timestamp-aware SGLang dataset when simulator validation must preserve the
original arrival process. Burst traffic and timestamp replay can produce different
scheduler order and prefix-cache reuse, so they should not be treated as the same
accuracy baseline.

## Output contract

Each scheduler rank writes a rank-qualified set. A TP-only run uses the prefix
`TP0`; data parallel, pipeline parallel, and expert parallel ranks add `-DPn`,
`-PPn`, and `-EPn` when their size is greater than one.

```text
TP0.manifest.json
TP0.request.jsonl
TP0.raw_request.jsonl
TP0.schedule_batch.jsonl
metrics.json                  # written by the serving benchmark
```

- `request.jsonl` is the normalized simulator trace: request timestamp, lengths,
  and device/host/storage prefix-hit lengths.
- `raw_request.jsonl` retains request identifiers, token IDs, queue timestamps,
  token latencies, and the unnormalized cache-hit counters.
- `schedule_batch.jsonl` records batch composition and synchronized
  `iter_latency` in seconds. This is the ML trainer input.
- `manifest.json` versions the schema and records row counts and timing semantics.

Do not commit collected data or trained model files to the SGLang repository.
