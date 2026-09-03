# Fun-ASR-Nano 显式 Intel XPU 诊断

本 POC 是 Intel XPU 路径的可选深度诊断。它对本条命令保持失败关闭：九项检查任一失败
即返回非零，绝不在诊断内部静默切换后端；但它**不能阻断 CUDA 或 CPU 初始化**，也不再是
`scripts/initialize-first-run.sh` 的全局安装硬门。日常首次初始化由 backend probe 为实际
候选完成张量与模型 smoke test，并把通过的结果写入私有 `selection.json`。

当前 Nano 运行时固定为 `native_funasr_pytorch`：通过本地 FunASR
`AutoModel(..., device="xpu:0")` 调用 PyTorch XPU。此前的 vLLM XPU
prompt-embedding 路径在本机连续第二次请求会卡死，已不再是可部署后端；旧报告不能作为
当前版本的放行证据。

> 显式诊断失败后，本条命令不会自行改走 CPU 或其他后端；需要日常可用性时，另行执行
> `scripts/initialize-first-run.sh` 的 auto 或显式 CPU/CUDA 路径。

## 硬门清单

`fun-voice-preflight` 按顺序执行九项检查,全部 `pass` 才返回 `ready=true`:

| 检查 | 断言 |
| --- | --- |
| `xpu_visible` | `torch.xpu.is_available()` 为真 |
| `nano_decoder_xpu` | 原生 FunASR/PyTorch Nano 解码器设备类型为 `xpu`、`backend == "native_funasr_pytorch"`，且能完成一次 XPU 显存分配 |
| `nano_encoder_xpu` | `next(audio_encoder.parameters()).device.type == "xpu"` |
| `nano_adaptor_xpu` | `next(audio_adaptor.parameters()).device.type == "xpu"` |
| `prompt_embeddings_xpu` | `next(embed_tokens.parameters()).device.type == "xpu"` |
| `decode_10s` | 10s 中英混合样本经 VAD 分段后，每段都有且仅有一个 Nano 结果，最终产出非空文本 |
| `decode_60s` | 60s 样本经 VAD 分段(≥2 段、按时间序)后，每段都有且仅有一个 Nano 结果，最终产出非空文本 |
| `no_cpu_decoder_fallback` | 引擎无 CPU 回退标志,解码器设备类型不是 `cpu` |
| `oom_survives` | 触发 OOM 后再次解码短样本成功,worker 进程仍可服务 |

加载约定为本地 FunASR `AutoModel`：`model=<本地 Nano snapshot>`、
`device="xpu:0"`、`trust_remote_code=True`、`disable_update=True`。加载器必须检查完整
Nano 参数、audio encoder、adaptor 和 prompt embedding 均位于 XPU；任何 CPU 参数直接
失败。原生路径没有 vLLM KV cache；旧配置中的 `gpu_memory_utilization`、`max_model_len`
与 `enforce_eager` 只为兼容旧配置保留，运行时明确忽略，不能用于内存调优。

## 成功证据格式

报告写入 `${XDG_RUNTIME_DIR}/fun-voice-ryan/poc-report.json`,只含 check 名、status、
detail 指标与样本构成(来源 URL、语言与时长),以 `0600` 权限保存；**不含音频路径与转写文本**。示意:

```json
{
  "device": "xpu:0",
  "ready": true,
  "checks": [
    {"name": "xpu_visible", "status": "pass", "detail": {"available": true}},
    {"name": "nano_decoder_xpu", "status": "pass", "detail": {
      "backend": "native_funasr_pytorch", "decoder_device_type": "xpu", "alloc_probe": "ok",
      "memory_before": {"allocated": 0, "reserved": 0},
      "memory_after": {"allocated": 1048576, "reserved": 67108864},
      "total_memory": 8589934592}},
    {"name": "decode_10s", "status": "pass", "detail": {"segment_count": 1, "text_length": 112}},
    {"name": "decode_60s", "status": "pass", "detail": {"segment_count": 2, "text_length": 701}},
    {"name": "oom_survives", "status": "pass", "detail": {
      "allocator_oom": "OutOfMemoryError",
      "recovery_text_length": 42}}
  ],
  "sample_composition": {
    "short": {"duration_s": 10.0, "sources": [{"source": "…asr_example_zh.wav", "language": "mandarin", "duration_s": 5.55}]},
    "long": {"duration_s": 60.0, "sources": []}
  }
}
```

成功判据:JSON 顶层 `ready == true`,九项 `status == "pass"`。
`memory_before` / `memory_after` 分别记录 1 MiB 探测分配前与分配中(尚未释放)的显存占用。
`decode_10s` / `decode_60s` 的 detail 只含 `segment_count`(VAD 分段数)与 `text_length`
(拼接文本长度),**不含转写文本**。

## 解码路径:VAD 分段 → 时间序拼接

`decode_10s` 与 `decode_60s` 不再对整段音频调用一次 `engine.generate()`,而是复用
`nano_runtime.FsmnVadSegmenter`(XPU FSMN-VAD)走与 worker 一致的分段路径:

1. 16 kHz 单声道音频加载为 float32 样本;
2. FSMN-VAD 输出语音段 `(start_ms, end_ms)` 列表;
3. 按 `start_ms` 时间序排序(不信任 VAD 返回顺序);
4. 每段边界加固定小重叠(`VAD_OVERLAP_MS = 250 ms`,与 `nano_runtime` 一致)后切片;
5. 逐段(一个 batch)送 Nano 推理;
6. 按段顺序**直接拼接**文本(不插入/删除任何字符)。

硬门第 5 条断言:`decode_10s` / `decode_60s` 文本非空;`decode_60s` 的 `segment_count`
≥ 2 且段按时间有序;Nano 返回数必须与切片数完全一致，每项均含字符串 `text`；拼接文本
按段顺序(`check_decode` 的 `min_segments` 参数强制执行)。

### 60s 样本构成

为让 60s 样本在 FSMN-VAD 下产生多个段,`run-nano-xpu-poc.sh` 在拼接相邻开源片段之间
插入 **0.4 s 静音**(≥0.3 s;无缝拼接会被 VAD 判成单段)。样本仍由普通话/英文开源片段
循环拼接成 60s,仅片段之间以静音分隔。

## 失败分类

| 失败检查 | 含义 | 排查方向 |
| --- | --- | --- |
| `xpu_visible` | `torch.xpu` 不可用 | Intel 驱动/Level Zero 未装或版本过旧;torch 是 CUDA/CPU 变体而非 `+xpu` |
| `nano_decoder_xpu` | 原生 Nano 未完整落到 XPU，或报告来自过期后端 | 检查本地 snapshot、FunASR/PyTorch XPU 安装及完整参数设备；重新运行当前 POC |
| `nano_encoder_xpu` / `nano_adaptor_xpu` / `prompt_embeddings_xpu` | FunASR 音频组件不在 xpu | `device` 传参错误;组件加载后未 `.to("xpu:0")` |
| `decode_10s` / `decode_60s` | 推理失败 | 模型不完整、tokenizer/权重缺失或 XPU OOM |
| `no_cpu_decoder_fallback` | 检测到 CPU 回退 | **禁止**;修复 FunASR/PyTorch XPU 环境,不得接受 |
| `oom_survives` | OOM 后进程无法继续服务 | worker 未捕获 OOM 或显存未释放;不得改走 CPU 规避 |

## Intel 驱动 / Level Zero 核验命令

```bash
# GPU 与 render 节点(需能看到 card0 + renderD128)
ls -l /dev/dri

# Level Zero / Intel compute runtime 包
dpkg -l | grep -iE 'level-zero|libze|intel-gpu' || true
ls /usr/lib/x86_64-linux-gnu/libze_loader.so* 2>/dev/null || true

# torch XPU 自检(环境内)
.venv/bin/python - <<'PY'
import torch
print("xpu_available:", torch.xpu.is_available())
if torch.xpu.is_available():
    print("device_count:", torch.xpu.device_count())
    print("device_name:", torch.xpu.get_device_name(0))
    print("total_memory:", torch.xpu.get_device_properties(0).total_memory)
PY

# Vulkan ICD(Intel 驱动提供的 Vulkan 运行时,llama.cpp/Vulkan 路线备查)
ls /usr/share/vulkan/icd.d/ 2>/dev/null || true

# VA-API / OpenCL(可选)
vainfo 2>/dev/null | head -5 || true
clinfo 2>/dev/null | grep -iE 'device name|version' | head -5 || true
```

Level Zero 是 torch-xpu 的底层运行时;若 `torch.xpu.is_available()` 为
`False`,优先核验 `libze-intel-gpu1` 与内核 i915 驱动是否就绪。

## 失败后的处置边界

- 脚本失败只记录失败分类并退出非零;**不会**、也不允许自动改走 CPU、CUDA 或更换后端。
- 是否研究 llama.cpp / Vulkan 替代路线是**人工决定**,不在自动化脚本的权限内。

## POC 结果

状态:**已通过（真实运行）** —— 当前运行态报告
`${XDG_RUNTIME_DIR}/fun-voice-ryan/poc-report.json` 为 `ready=true`，设备为
`xpu:0`，九项硬门全部为 `pass`；Nano decoder 后端为
`native_funasr_pytorch`。该报告以 `0600` 保存，只记录设备、门禁指标和样本构成，
不含音频路径或转写文本，可作为当前 Intel XPU 路径的补充诊断证据。

下表摘录的是该真实运行报告的非敏感指标。任何后续环境、驱动或模型变更后，都必须重新运行
本节末的 POC；新报告只有仍满足 `ready=true`、九项全 `pass` 且
`backend="native_funasr_pytorch"` 时才表示该显式诊断通过。

### 九项硬门结果

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| `xpu_visible` | pass | `torch.xpu.is_available() == True` |
| `nano_decoder_xpu` | pass | `backend=native_funasr_pytorch`、`decoder_device_type=xpu`、`alloc_probe=ok` |
| `nano_encoder_xpu` | pass | audio_encoder 参数为 `xpu` |
| `nano_adaptor_xpu` | pass | audio_adaptor 参数为 `xpu` |
| `prompt_embeddings_xpu` | pass | embed_tokens 参数为 `xpu` |
| `decode_10s` | pass | VAD 分段 1 段；结果非空（长度指标 20） |
| `decode_60s` | pass | VAD 分段 2 段、按时间序拼接；结果非空（长度指标 508） |
| `no_cpu_decoder_fallback` | pass | `decoder_device_type=xpu`，无 CPU 回退原因 |
| `oom_survives` | pass | `allocator_oom=OutOfMemoryError`，恢复解码成功 |

报告路径:`${XDG_RUNTIME_DIR}/fun-voice-ryan/poc-report.json`。

### 解码检查已走 VAD 分段路径

早期 POC 的 `decode_10s` / `decode_60s` 直接对整段音频调用一次
`engine.generate()`(`result_count == 1`),不满足硬门第 5 条"VAD 分段与时间序拼接"。
现已改为复用 `FsmnVadSegmenter`:VAD 分段 → 按 `start_ms` 排序 → 固定重叠切片 →
逐段 Nano 推理 → 按段序直接拼接,并拒绝任何结果数不等于 VAD 段数或缺少字符串 `text` 的
响应。报告 detail 以 `segment_count` + `text_length` 记录证据(见上文"解码路径")；
文本长度可能随模型采样而变化，以最新 JSON 报告为准。


### 复现

```bash
./scripts/create-xpu-env.sh          # 建原生 FunASR/PyTorch XPU 环境
./scripts/run-nano-xpu-poc.sh        # 下载样本/模型,跑九项硬门,输出 poc-report.json
```

单测(9 项硬门逻辑,假 torch/原生 Nano):`.venv/bin/pytest tests/test_preflight.py -q`。
