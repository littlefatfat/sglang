# 验证记录

## 环境门槛

```text
SGLang: v0.5.16 基线或当前 v0.5.16 适配分支
transformers: 5.12.1
sgl-kernel: 0.4.5
simulation mode: OFFLINE
```

检查：

```bash
python3 scripts/check_environment.py
```

## v0.5.16 最终功能验收

日期：2026-07-29。命令：

```bash
HISIM_ACCEPTANCE_DIR=/data2/maruiyan.mry/hisim-sglang/validation/v0516/final-acceptance \
  bash scripts/acceptance.sh
```

结果：

```text
bundle tests: 10 passed
simulator unit/hook/predictor tests: 11 passed
in-process OFFLINE replay: 3/3
in-process BLOCKING AIC random: 2/2
in-process OFFLINE AIC ShareGPT: 2/2
Qwen3-32B-FP8 H20 AIC: 3/3
GLM5 P-node B300 TP8 AIC: 3/3
DSv4Pro P-node GB300 TP4 ML: 3/3
service OFFLINE trace replay: 3/3
service BLOCKING terminal random: 2/2
service BLOCKING terminal ShareGPT: 2/2
CPU Python page allocator: pass
GPU A100 Triton page allocator: pass
PASS all acceptance checks
```

关键结果：

| Case | Mean TTFT | Mean E2E | Prefix reused |
|---|---:|---:|---:|
| replay in-process | 8.583 ms | 8.583 ms | 0.250000 |
| BLOCKING AIC in-process | 32.128 ms | 32.128 ms | 0 |
| ShareGPT AIC in-process | 36.916 ms | 129.533 ms | 0 |
| GLM5 P-node AIC | 31.862 ms | 31.862 ms | 0 |
| DSv4Pro P-node ML | 290.666 ms | 290.666 ms | 0 |
| service OFFLINE replay | 8.531 ms | 8.531 ms | 0.250000 |
| service BLOCKING ShareGPT | 10.947 ms | 86.302 ms | 0.642857 |

验收过程中修复了 v0.5.16 benchmark CLI 入口、BLOCKING client pacing、
BLOCKING wall/simulation clock 混用、服务 warmup readiness 竞态、服务模式
request 文件计数，以及 Qwen H20 AIC database version。最终目录含 `PASS`
哨兵和 21 份分项日志。

## 代码回归

```bash
bash scripts/test_bundle.sh
pytest -q ../test/test_simulation_sglang_runner.py
bash scripts/validate_cpu_gpu.sh
```

验收：

```text
bundle tests: 10 pass
simulator unit/hook/predictor tests: 11 pass
deterministic runner: pass
CPU-only runner + trace replay: pass
GPU-visible runner + trace replay: pass
```

确定性 runner 必须覆盖：

```text
cold -> L1 device hit -> L2 host hit -> L3 storage hit
TTFT/E2E >= 0
request count exact
```

CPU/GPU 验证定义：

```text
CPU-only: CUDA_VISIBLE_DEVICES=""，torch.cuda.is_available() == false
GPU-visible: 指定物理 GPU，torch.cuda.is_available() == true
CPU spawned runner 覆盖 cold -> L1 -> L2 -> L3
CPU/GPU 均执行 page_size=256 的同一份 replay trace
```

GPU-visible replay 使用 `device=cuda`，因此
`PagedTokenToKVPoolAllocator.alloc_extend/alloc_decode` 实际调用 v0.5.16
的 Triton kernel。`load_format=dummy` 不加载模型权重，HiSim hook 不执行
真实模型 forward；step 时间仍来自所选 predictor。

2026-07-28 实跑：

```text
container: hisim-v0516-dev
CPU-only: torch.cuda.is_available() == false
GPU-visible: NVIDIA A100-SXM4-80GB
CPU spawned runner: pass (cold -> L1 -> L2 -> L3)
GPU-visible spawned runner: pass
CPU page_size=256 replay: 3/3 requests, pass
GPU page_size=256 replay: 3/3 requests, pass
```

v0.5.16 没有删除 GPU page allocator 优化：

```text
python/sglang/kernels/ops/memory/allocator.py
PagedTokenToKVPoolAllocator.alloc_extend -> alloc_extend_kernel (Triton)
PagedTokenToKVPoolAllocator.alloc_decode -> alloc_decode_kernel (Triton)
```

当前 kernel 还包含 `free_page_ptr` 的 `do_not_specialize` 和动态 blocked loop，
用于减少第二种指针对齐产生的额外 JIT，以及避免按 extend size 展开。

注意：GPU 首次运行包含 CUDA/Triton 初始化和 JIT，且当前实现会把这段
wall-clock 计入仿真 `cpu_overhead`。因此 CPU/GPU smoke 只做功能验收；
准确度或性能对比必须预热，并使用固定/回放 CPU overhead。

## 历史实测基线

来源：<https://github.com/sgl-project/sglang/issues/21891>。

Qwen3-8B / H20：

| Case | TTFT MAPE | TPOT MAPE | ITL MAPE | Input throughput MAPE | Duration MAPE | Prefix hit MAPE |
|---|---:|---:|---:|---:|---:|---:|
| no-cache | 3.42% | 1.64% | 1.78% | 1.41% | 1.39% | 0.00% |
| L1 | 4.15% | 3.47% | 3.54% | 2.40% | 2.33% | 0.04% |
| L2 | 2.38% | 4.05% | 4.07% | 2.35% | 2.29% | 0.00% |

Qwen3-32B-FP8 / H20：

| Case | TTFT MAPE | TPOT MAPE | ITL MAPE | Input throughput MAPE | Duration MAPE | Prefix hit MAPE |
|---|---:|---:|---:|---:|---:|---:|
| no-cache | 2.77% | 0.58% | 0.52% | 0.52% | 0.51% | 0.00% |
| L1 | 2.40% | 1.03% | 1.02% | 1.04% | 1.03% | 0.04% |
| L2 | 3.05% | 1.11% | 1.02% | 1.13% | 1.12% | 0.00% |

## v0.5.16 Replay 回归

同一 trace、同一 `server_args`、同一 replay table：

```bash
python3 scripts/compare_results.py \
  --old <0714-result-dir> \
  --new <v0516-result-dir>
```

必须报告：

```text
prefix/device/host/storage hit delta
iteration count
batch-size histogram
exact batch composition sequence
first mismatch iteration
TTFT/E2E APE
```

框架准确度门槛：

```text
prefix hit delta = 0
batch composition sequence = exact match
request count = exact match
```

时间准确度门槛：

```text
Replay miss rate near 0 后再判断 TTFT/E2E
ML step MAPE <= 5% 才进入 E2E 验证
AIC 必须单独报告 SOL/SILICON 与实测误差
```

## GLM5 v0.5.16 实跑

日期：2026-07-28。Case：

```text
hisim-num-node-1-glm-5-blksz-256-bucket-85-128-cnt-1816-time-60min-pod-9p7wt_slowdown_factor_1
1816 requests / 60 minutes / GLM-5.1-FP8 / B300 / TP8 replay config
```

旧版与 v0.5.16 使用相同 trace、`server_args` 和纯 forward replay table：

| 指标 | 旧版 | v0.5.16 | 相对误差 |
|---|---:|---:|---:|
| Prefix reused ratio | 0.6699217 | 0.6700850 | 0.0244% |
| L1 device hit ratio | 0.6230461 | 0.6304010 | 1.1805% |
| L2 host hit ratio | 0.0468756 | 0.0396840 | 15.3419% |
| Mean TTFT/E2E | 2542.83 ms | 2657.92 ms | 4.5261% |
| P90 TTFT/E2E | 5702.74 ms | 5952.61 ms | 4.3817% |

Batch：

```text
old iterations: 4747
new iterations: 4582
old/new mean batch size: 1.1106 / 1.1910
first mismatch iteration: 5 (zero-based)
exact sequence match: false
```

前五步 replay forward latency 完全一致。分叉前，旧版每步记录的
preprocess+postprocess 约 39–44 ms；v0.5.16 将本机 wall-clock
`cpu_overhead` 约 57 ms 推进仿真时钟，五步累计差约 80 ms，使下一请求进入
第六个 batch。`0714_mry_dev` 包含这段 wall-clock 推进，
`refactor-simplify` 删除了它。

结论：

```text
前缀树总命中量稳定；L1/L2 分层和 batch 时序未保持不变。
偏差源是仿真机 wall-clock CPU overhead，不是 replay predictor 或前缀树。
OFFLINE 准确度回归不能继续依赖本机实测 CPU overhead。
后续应使用固定/回放 CPU overhead，或使用包含 pre/post 的 replay table 并禁用重复计时。
```

## DSv4Pro v0.5.16 实跑

日期：2026-07-28。Case：

```text
hisim-num-node-1-dpskv4pro-blksz-256-bucket-0-32-cnt-2304-time-60min-pod-7s07-008
2304 requests / 60 minutes / DeepSeek-V4-Pro / B300 / TP4 / L2 / HGB monotonic
real mean TTFT: 411.171455 ms
```

### 0714 语义与 v0.5.16 适配

| 指标 | 0714 历史结果 | v0.5.16 | 差值/相对变化 |
|---|---:|---:|---:|
| Prefix reused ratio | 0.4110452 | 0.4183055 | +0.7260 pp |
| L1 device hit ratio | 0.3640169 | 0.3749836 | +1.0967 pp |
| L2 host hit ratio | 0.0470283 | 0.0433219 | -0.3706 pp |
| Mean TTFT/E2E | 433.739 ms | 433.819 ms | +0.0185% |
| Median TTFT/E2E | 221.286 ms | 226.662 ms | +2.4291% |
| P90 TTFT/E2E | 1066.372 ms | 975.454 ms | -8.5260% |

相对真实 mean TTFT：

```text
0714 historical: +5.4886%
v0.5.16 with 0714 CPU-overhead semantics: +5.5081%
```

Batch：

```text
0714 / v0.5.16 iterations: 2585 / 2571
0714 / v0.5.16 mean batch size: 1.0766 / 1.0821
first mismatch iteration: 5 (zero-based)
exact sequence match: false
```

因此，这一组的 mean TTFT 保持稳定，总前缀命中差小于 1 个百分点；但
device/host 分层、batch composition 和 p90 没有保持不变，尚未通过严格的
仿真语义回归门槛。

### CPU overhead 消融

同一 v0.5.16 代码和输入，只切换：

```json
{"scheduler": {"ignore_cpu_overhead": true}}
```

| 指标 | 0714 语义（false） | 忽略 overhead（true） | 相对变化 |
|---|---:|---:|---:|
| Prefix reused ratio | 0.4183055 | 0.4183055 | 0.0000% |
| L1 device hit ratio | 0.3749836 | 0.3742860 | -0.0698 pp |
| L2 host hit ratio | 0.0433219 | 0.0440195 | +0.0698 pp |
| Mean TTFT/E2E | 433.819 ms | 373.905 ms | -13.8109% |
| Median TTFT/E2E | 226.662 ms | 204.695 ms | -9.6913% |
| P90 TTFT/E2E | 975.454 ms | 807.624 ms | -17.2053% |

相对真实 mean TTFT：

```text
0714 semantics: +5.5081%
ignore CPU overhead: -9.0623%
```

结论：

```text
refactor-simplify 删除 wall-clock CPU overhead 不是本 case 总前缀命中偏差的主因。
它会显著改变 TTFT、batch 边界和少量 L1/L2 分层。
默认必须保留 0714 语义；ignore_cpu_overhead 只作为显式 A/B 开关。
旧版到 v0.5.16 的 +0.726 pp 总前缀漂移仍需用多 trace、尤其高命中和多节点 case 定位。
单 case 不能证明 DSv4Pro 全量误差已经合格。
```

## 官方 v0.5.16 镜像全量验收

日期：2026-07-29。环境：

```text
image: lmsysorg/sglang:v0.5.16
sglang: 0.5.16 (/sgl-workspace/sglang/python)
torch: 2.11.0+cu130
numpy: 2.3.5
GPU: NVIDIA A100-SXM4-80GB
container: hisim-v0516-official-0729
service port: 31029
```

结果目录：

```text
/data2/maruiyan.mry/hisim-sglang/validation/v0516/official-image-0729-final-v6
```

`PASS` 文件存在。`acceptance.sh` 全部通过：29 个测试、服务化和同进程启动、
OFFLINE/BLOCKING、random/ShareGPT/trace、replay/AIC/ML、Qwen3-8B、
Qwen3-32B、GLM5 P 节点、DSv4Pro P 节点，以及纯 CPU allocator 和 GPU
Triton paged allocator。AIC 以 `--no-deps` 安装并在 NumPy 2.3.5 下完成真实
预测调用；未降级官方镜像的 NumPy。

Simulator 使用不带 `--no-deps` 的普通 editable 安装。pip 识别
`sglang==0.5.16`、NumPy 2.3.5、scikit-learn 1.8.0 和 joblib 1.5.3 均已满足；
安装前后四个版本完全一致。安装后的包元数据同时包含 SGLang 和三个
simulator 直接依赖。

机器学习 predictor 已统一为 `name=ml` 和固定18特征 ABI；运行时 `gbr`
实现及现场训练入口已删除。HGBMono、GBR 或其他 sklearn-compatible 回归器
统一通过离线 joblib bundle 接入。以下验证均通过：

- 任意带 `predict()` 的回归器可经 `ml.py` 预测并应用 `latency_scale`。
- 缺少 `model`/`features` 元数据，或特征名称、顺序不一致时加载直接失败。
- 当前 DSv4Pro HGBMono bundle 实际加载18特征并完成预测。
- 源码中不存在残留的运行时 `gbr` dispatcher 或实现引用。

Replay 只替换 forward latency，0714 CPU overhead 语义保持开启。验收中的
forward-only replay 为 3/3 exact match、0 fallback，且
`total_cpu_s=0.00518`，确认 CPU overhead 未被关闭。

### 服务化与 Python 同进程一致性

同一 Qwen3-8B 配置、同一三请求 trace、同一 forward-only replay table：

| 项目 | 结果 |
|---|---|
| request/token 数 | exact |
| batch composition | 3/3 exact |
| forward latency | exact |
| prefix/L1/L2 命中率 | exact |
| replay coverage | 3 exact / 0 miss |
| 同进程 CPU overhead | 5.194 ms |
| 服务化 CPU overhead | 4.608 ms |
| Mean TTFT | 8.732 ms / 8.537 ms |

两种启动方式的仿真语义一致。时间不做 bitwise exact：0714 语义使用本机
wall-clock CPU overhead，进程形态会带来亚毫秒差异。本次所有时间指标差异均
小于逐步 CPU overhead 差值总和 `0.603 ms`。

## DSv4Pro L2 BLOCKING 验证

使用 DSv4Pro P 节点历史 trace 的前 100 个请求，`timestamp_scale=1000`，
在官方 v0.5.16 GPU 镜像中运行 BLOCKING：

```text
result: PASS
completed: 100
iterations: 37
L2 host hit ratio: 0.0723926380
total L2 logical load: 0.652790218 s
total L2 blocked wall: 0.652944369 s
total CPU overhead: 0.265831189 s
```

实际阻塞与逻辑 load 相差 `0.154 ms`，属于 `sleep` 调度开销。该配置关闭
overlap schedule，因此全部有效 L2 load 都会阻塞。CPU overhead 在计算时
扣除了实际 L2 阻塞墙钟，没有重复加入这 `0.653 s`。

结果目录：

```text
/data2/maruiyan.mry/hisim-sglang/validation/v0516/blocking-l2-load
```
