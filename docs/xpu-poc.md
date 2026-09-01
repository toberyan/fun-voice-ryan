# Fun-ASR-Nano Intel XPU 阻断 POC

本 POC 是桌面服务上线的**硬门**:九项检查全部通过之前,不得安装或启动任何桌面服务;
任一失败即停止部署,绝不静默退回 CPU 或切换后端。

> 失败后,**只有人工**可以决定是否研究 llama.cpp / Vulkan 替代路线。脚本与自动化不得
> 在 POC 失败时自行改走 CPU 或其他后端。

## 硬门清单

`fun-voice-preflight` 按顺序执行九项检查,全部 `pass` 才返回 `ready=true`:

| 检查 | 断言 |
| --- | --- |
| `xpu_visible` | `torch.xpu.is_available()` 为真 |
| `vllm_xpu_decoder` | vLLM 解码器设备类型为 `xpu`,且能完成一次 XPU 显存分配 |
| `nano_encoder_xpu` | `next(audio_encoder.parameters()).device.type == "xpu"` |
| `nano_adaptor_xpu` | `next(audio_adaptor.parameters()).device.type == "xpu"` |
| `prompt_embeddings_xpu` | `next(embed_tokens.parameters()).device.type == "xpu"` |
| `decode_10s` | 10s 中英混合样本经 VAD 分段后，每段都有且仅有一个 Nano 结果，最终产出非空文本 |
| `decode_60s` | 60s 样本经 VAD 分段(≥2 段、按时间序)后，每段都有且仅有一个 Nano 结果，最终产出非空文本 |
| `no_cpu_decoder_fallback` | 引擎无 CPU 回退标志,解码器设备类型不是 `cpu` |
| `oom_survives` | 触发 OOM 后再次解码短样本成功,worker 进程仍可服务 |

加载约定沿用 FunASR 官方 `funasr/models/fun_asr_nano/inference_vllm.py`
(`FunASRNanoVLLM.from_pretrained`):`device="xpu:0"`、`dtype="bf16"`、
`tensor_parallel_size=1`、`gpu_memory_utilization=0.35`、`max_model_len=4096`、
`enforce_eager=True`。音频 encoder、adaptor、prompt embedding 由 FunASR 显式移动到
`xpu:0`。

## 成功证据格式

报告写入 `${XDG_RUNTIME_DIR}/fun-voice-ryan/poc-report.json`,只含 check 名、status、
detail 指标与样本构成(来源 URL、语言与时长),以 `0600` 权限保存；**不含音频路径与转写文本**。示意:

```json
{
  "device": "xpu:0",
  "ready": true,
  "checks": [
    {"name": "xpu_visible", "status": "pass", "detail": {"available": true}},
    {"name": "vllm_xpu_decoder", "status": "pass", "detail": {
      "decoder_device_type": "xpu", "alloc_probe": "ok",
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
`nano_runtime.FsmnVadSegmenter`(CPU FSMN-VAD)走与 worker 一致的分段路径:

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
| `vllm_xpu_decoder` | vLLM 未落到 XPU | `vllm-xpu-kernels` 未装;Python 非 3.12;平台自动检测失败 |
| `nano_encoder_xpu` / `nano_adaptor_xpu` / `prompt_embeddings_xpu` | FunASR 音频组件不在 xpu | `device` 传参错误;组件加载后未 `.to("xpu:0")` |
| `decode_10s` / `decode_60s` | 推理失败 | 模型不完整、tokenizer/权重缺失、KV cache OOM |
| `no_cpu_decoder_fallback` | 检测到 CPU 回退 | **禁止**;vLLM 平台检测回退到 CPU,需修复环境,不得接受 |
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

Level Zero 是 torch-xpu / vLLM XPU 的底层运行时;若 `torch.xpu.is_available()` 为
`False`,优先核验 `libze-intel-gpu1` 与内核 i915 驱动是否就绪。

## 失败后的处置边界

- 脚本失败只记录失败分类并退出非零;**不会**、也不允许自动改走 CPU、CUDA 或更换后端。
- 是否研究 llama.cpp / Vulkan 替代路线是**人工决定**,不在自动化脚本的权限内。

## POC 结果

状态:**DONE** —— 九项硬门全部通过(`ready=true`,退出码 0),Fun-ASR-Nano 在
Intel Arc(Arc 130T/140T,Arrow Lake-P iGPU)上通过 vLLM XPU 后端完成真实解码。
**未**退回 CPU,未更换后端。

### 九项硬门结果

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| `xpu_visible` | pass | `torch.xpu.is_available() == True` |
| `vllm_xpu_decoder` | pass | decoder_device_type=xpu,alloc_probe=ok |
| `nano_encoder_xpu` | pass | audio_encoder 参数在 xpu |
| `nano_adaptor_xpu` | pass | audio_adaptor 参数在 xpu |
| `prompt_embeddings_xpu` | pass | embed_tokens 参数在 xpu |
| `decode_10s` | pass | VAD 分段后每段对应一个 Nano 结果，产出非空文本 |
| `decode_60s` | pass | VAD 分段≥2、按时间序拼接，且结果数与段数完全一致 |
| `no_cpu_decoder_fallback` | pass | decoder_device_type=xpu,无回退 |
| `oom_survives` | pass | allocator_oom=OutOfMemoryError,恢复解码成功 |

报告路径:`${XDG_RUNTIME_DIR}/fun-voice-ryan/poc-report.json`。

### 解码检查已走 VAD 分段路径

早期 POC 的 `decode_10s` / `decode_60s` 直接对整段音频调用一次
`engine.generate()`(`result_count == 1`),不满足硬门第 5 条"VAD 分段与时间序拼接"。
现已改为复用 `FsmnVadSegmenter`:VAD 分段 → 按 `start_ms` 排序 → 固定重叠切片 →
逐段 Nano 推理 → 按段序直接拼接,并拒绝任何结果数不等于 VAD 段数或缺少字符串 `text` 的
响应。报告 detail 以 `segment_count` + `text_length` 记录证据(见上文"解码路径")；
文本长度可能随模型采样而变化，以最新 JSON 报告为准。


### 环境版本

```
torch==2.13.0+xpu   torchaudio==2.11.0+xpu   torchvision==0.28.0+xpu
vllm==0.28.0        vllm-xpu-kernels==0.1.14.1
triton==3.7.2+xpu   triton-xpu==3.7.2        oneccl==2022.0.0
funasr==1.4.11 (git@8cd758c0ced576516b05a749194e6a94cdd38f99)   modelscope==1.39.1
```

### 应用的兼容修复(create-xpu-env.sh 内,幂等)

1. **安装 `vllm-xpu-kernels==0.1.14.1`**:vLLM 0.28.0 通用 wheel 不声明该依赖。
2. **oneCCL 单卡 warm-up 修复(vllm#52386 / PR#52389)**:仅多卡才 warm-up。
3. **显存探测回退**:驱动 25.18(< 26.18)`getMemoryInfo` 返回 0,回退为
   `total - reserved`。
4. **triton XPU shim 固定**:卸载 PyPI `triton==3.8.0`(NVIDIA-only,由 xgrammar
   的 `Requires-Dist: triton` 拉入),改装 wheels.vllm.ai/xpu 的
   `triton==3.7.2+xpu` shim(Requires-Dist `triton-xpu==3.7.2`),透明解析到真正
   的 Intel XPU 实现。
5. **Level Zero 头文件 + 链接软链**:triton-xpu Intel 后端 JIT 编译 driver.c 需要
   `<level_zero/ze_api.h>` 与 `libze_loader.so`(由 level-zero-dev 提供,本机无且
   不能 sudo);从 oneapi-src/level-zero v1.21.9(匹配 libze1 1.21.9)下载头文件
   并入 `.venv/include/level_zero/`,并软链 `.venv/lib/libze_loader.so`。

### preflight 内的两项适配(src/fun_voice/preflight.py)

- **`attention_backend=TRITON_ATTN`**:vllm-xpu-kernels 0.1.14.1 的 CUTLASS
  FlashAttention 只有 XE2/XE3 内核,Arc 130T/140T(device_id 0x7D51,Xe-LPG+)
  报 `Only XE2/XE3 cutlass kernel is supported currently`;改用 Triton 注意力
  后端(依赖第 4/5 项修复的 triton Intel 后端)。
- **OOM 探测改为分配器 OOM**:vLLM 0.28.0 不拒绝 `max_tokens > max_model_len`(只
  截断),大 `max_tokens` 在长解码后会令 V1 调度器挂起;`oom_survives` 改用
  `torch.empty` 超总量触发 `OutOfMemoryError` 作为可靠 OOM,再验证恢复解码。

### 复现

```bash
./scripts/create-xpu-env.sh          # 建环境(含五项兼容修复)
./scripts/run-nano-xpu-poc.sh        # 下载样本/模型,跑九项硬门,输出 poc-report.json
```

单测(9 项硬门逻辑,假 torch/vLLM/Nano):`.venv/bin/pytest tests/test_preflight.py -q`。
