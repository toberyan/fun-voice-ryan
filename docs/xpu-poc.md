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
| `decode_10s` | 10s 中英混合样本解码产出非空文本 |
| `decode_60s` | 60s 样本解码产出非空文本 |
| `no_cpu_decoder_fallback` | 引擎无 CPU 回退标志,解码器设备类型不是 `cpu` |
| `oom_survives` | 触发 OOM 后再次解码短样本成功,worker 进程仍可服务 |

加载约定沿用 FunASR 官方 `funasr/models/fun_asr_nano/inference_vllm.py`
(`FunASRNanoVLLM.from_pretrained`):`device="xpu:0"`、`dtype="bf16"`、
`tensor_parallel_size=1`、`gpu_memory_utilization=0.35`、`max_model_len=4096`、
`enforce_eager=True`。音频 encoder、adaptor、prompt embedding 由 FunASR 显式移动到
`xpu:0`。

## 成功证据格式

报告写入 `${XDG_RUNTIME_DIR}/fun-voice-ryan/poc-report.json`,只含 check 名、status、
detail 指标与样本构成(来源 URL 与时长),**不含音频路径与转写文本**。示意:

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
    {"name": "decode_10s", "status": "pass", "detail": {"result_count": 1, "text_length": 42}},
    {"name": "oom_survives", "status": "pass", "detail": {
      "oversized_request_error": "ValueError",
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

状态:**DONE_WITH_CONCERNS** —— 环境、模型加载、XPU 平台识别、音频组件上卡均已
验证通过,但 vLLM 引擎在最后一步(Triton 内核 warm-up)因 Intel triton-xpu 生态
打包不完整而失败。**未**退回 CPU,未更换后端。

### 已通过

- `torch.xpu.is_available() == True`;设备 `Intel(R) Arc(TM) Graphics`,可寻址显存
  28.48 GiB;XPU 张量运算正常。
- vLLM 0.28.0 识别 XPU 平台(`current_platform.device_type == "xpu"`),成功解析
  模型架构 `Qwen3ForCausalLM` 并初始化 V1 引擎。
- FunASR 加载 `audio_encoder`(914 参数)与 `audio_adaptor`(36 参数)至 `xpu:0`。
- 三项环境兼容修复已应用并验证(见下)。

### 环境版本

```
torch==2.13.0+xpu   torchaudio==2.11.0+xpu   torchvision==0.28.0+xpu
vllm==0.28.0        vllm-xpu-kernels==0.1.14.1
triton==3.8.0       triton-xpu==3.7.2        oneccl==2022.0.0
funasr==1.4.11 (git@8cd758c0ced576516b05a749194e6a94cdd38f99)   modelscope==1.39.1
```

### 应用的三项兼容修复(create-xpu-env.sh 内,幂等)

1. **安装 `vllm-xpu-kernels==0.1.14.1`**:vLLM 0.28.0 通用 wheel 不声明该依赖,缺
   少时 `vllm.platforms.xpu` 无法导入,`current_platform.device_type` 为空。
2. **oneCCL 单卡 warm-up 修复(vllm#52386 / PR#52389)**:`world_size==1` 仍做
   `all_reduce` warm-up,本机 oneCCL 无法枚举 GPU 拓扑时报
   `ze_data was not initialized`;补丁仅多卡才 warm-up。
3. **显存探测回退**:驱动 25.18(< 26.18)`getMemoryInfo` 返回 0 可用显存,启动检查
   误判"显存不足";补丁回退为 `total - reserved`。

### 剩余阻塞(失败分类:driver/triton 兼容)

引擎在 `kernel_warmup` 阶段失败:

```text
TypeError: 'function' object is not subscriptable
vllm/v1/worker/block_table.py:195  _compute_slot_mapping_kernel[(num_reqs + 1,)](
```

根因:`triton-xpu==3.7.2`(torch 2.13.0+xpu 强制依赖)只提供 Intel 后端扩展
(`triton/backends/intel/`),其 `compiler.py` 需要 `triton._C.libtriton.intel`,但
PyPI 的 `triton==3.8.0`(vLLM 0.28.0 依赖)编译时不含 Intel 后端;`triton==3.7.2+xpu`
兼容 shim 又不提供任何模块。于是 vLLM 禁用 Triton(`Disabling Triton`),而其内部
Triton 内核(block table slot mapping)仍被引用,报 `function is not subscriptable`。
本机 Intel 驱动为 compute-runtime 25.18,低于 vLLM-XPU 文档建议的 26.18(升级需
sudo,本任务禁止系统层装包)。

**结论**:在保持 XPU(不退回 CPU)的前提下,剩余阻塞是 Intel triton-xpu 打包与驱动
版本缺口。是否研究 llama.cpp / Vulkan 替代路线,**由人工决定**,脚本不会自行切换。

### 复现

```bash
./scripts/create-xpu-env.sh          # 建环境(含三项兼容修复)
./scripts/run-nano-xpu-poc.sh        # 下载样本/模型,跑九项硬门,输出 poc-report.json
```

单测(9 项硬门逻辑,假 torch/vLLM/Nano):`uv run pytest tests/test_preflight.py -q`。
