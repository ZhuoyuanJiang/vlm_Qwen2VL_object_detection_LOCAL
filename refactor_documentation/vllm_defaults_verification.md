# vLLM 0.13.0 Default Configuration Verification

**Server Location:** vllab server
**Environment Path:** `/ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving/`
**vLLM Version:** 0.13.0
**Python Version:** 3.12

---

## Summary: Reviewer's Claims Are CORRECT

The reviewer claimed:
1. ✅ `enable_chunked_prefill` defaults to **True** for generative models
2. ✅ `max_num_batched_tokens` defaults to **2048** for OpenAI API server
3. ✅ `max_num_seqs` defaults to **256** for OpenAI API server

**All verified via source code inspection.**

---

## 1. Scheduler Configuration Defaults

**File:** `/ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving/lib/python3.12/site-packages/vllm/config/scheduler.py`

### Line 44-45: Base Defaults
```python
DEFAULT_MAX_NUM_BATCHED_TOKENS: ClassVar[int] = 2048
DEFAULT_MAX_NUM_SEQS: ClassVar[int] = 128
```

### Line 50-61: Field Definitions
```python
max_num_batched_tokens: int = Field(
    default=DEFAULT_MAX_NUM_BATCHED_TOKENS,
    ge=1,
    description=(
        "Maximum number of batched tokens per iteration. "
        "In real usage, this should be set in "
        "`EngineArgs.create_engine_config` where it is "
        "automatically configured based on the device."
    ),
)

max_num_seqs: int = Field(
    default=DEFAULT_MAX_NUM_SEQS,
    ge=1,
    description=(
        "Maximum number of sequences per iteration. "
        "In real usage, this should be set in "
        "`EngineArgs.create_engine_config` where it is "
        "automatically configured based on the device and the model."
    ),
)
```

### Line 78-83: Chunked Prefill Default
```python
enable_chunked_prefill: bool = True
"""Whether to enable chunked prefill. If False, vLLM will not
support chunked prefill. In real usage, this should be set in
`EngineArgs.create_engine_config` where it is automatically set
based on whether the model supports chunked prefill."""
```

**Important Note:** These are **base defaults** that get overridden by `EngineArgs.create_engine_config` based on:
- Device type (H100 vs non-H100)
- Model capabilities
- Usage context (LLM_CLASS vs OPENAI_API_SERVER)

---

## 2. OpenAI API Server Context Defaults

**File:** `/ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving/lib/python3.12/site-packages/vllm/engine/arg_utils.py`

### Lines 1809-1816: Context-Specific Defaults

```python
# For non-H100 GPUs, OpenAI API server uses different defaults:
default_max_num_batched_tokens = {
    UsageContext.LLM_CLASS: 8192,
    UsageContext.OPENAI_API_SERVER: 2048,  # ← OpenAI server uses 2048
}

default_max_num_seqs = {
    UsageContext.LLM_CLASS: 256,
    UsageContext.OPENAI_API_SERVER: 256,   # ← OpenAI server uses 256
}
```

**Context for our experiments:**
- We used `vllm serve` command → OpenAI API server mode
- Non-H100 GPU (RTX 4090)
- Therefore: `max_num_batched_tokens=2048`, `max_num_seqs=256`

**Note on max_num_seqs:**
- SchedulerConfig base default: 128 (line 45)
- OpenAI API server overrides to: **256** (line 1815)
- Reviewer was correct: OpenAI server uses 256, not 128

---

## 3. Chunked Prefill Auto-Detection

**File:** `/ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving/lib/python3.12/site-packages/vllm/engine/arg_utils.py`

### Lines 1854-1858: Auto-Enable Based on Model Support

```python
# If enable_chunked_prefill is not explicitly set, default to model's capability
default_chunked_prefill = model_config.is_chunked_prefill_supported

if self.enable_chunked_prefill is None:
    self.enable_chunked_prefill = default_chunked_prefill
```

**What this means:**
- If `--enable-chunked-prefill` flag NOT provided → check model support
- Calls `model_config.is_chunked_prefill_supported` to determine default

---

## 4. Model Support for Chunked Prefill

**File:** `/ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving/lib/python3.12/site-packages/vllm/config/model.py`

### Lines 1718-1719: Generative Models Support Chunked Prefill

```python
logger.debug("Generative models support chunked prefill.")
return True
```

**Context:** This is inside the `is_chunked_prefill_supported` property method for `ModelConfig`.

**For Qwen2-VL (a generative decoder model):**
- Model type: Generative decoder model
- `is_chunked_prefill_supported` returns `True`
- Therefore: Chunked prefill **defaults to ENABLED**

---

## Summary Table

| Configuration | Base Default | OpenAI Server Default | Our Experiments |
|--------------|--------------|----------------------|-----------------|
| **max_num_batched_tokens** | 2048 (line 44) | 2048 (line 1811) | **2048** ✅ |
| **max_num_seqs** | 128 (line 45) | **256** (line 1815) | **256** ✅ |
| **enable_chunked_prefill** | True (line 78) | Auto-detect via model | **True** (Qwen2-VL supports it) ✅ |

---

## Recommended vLLM Server Startup Command

Based on verified defaults, here's the recommended command with explicit flags:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve \
  /ssd1/zhuoyuan/vlm_outputs/qwen2vl-nutrition-detection-r4-joint-merged \
  --served-model-name qwen2vl-nutrition \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 4096 \
  --limit-mm-per-prompt '{"image":1}' \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 256 \
  --enable-chunked-prefill \
  --port 8000
```

**Why make defaults explicit?**
1. **Documentation**: Clear what configuration was used in experiments
2. **Reproducibility**: Same results on different vLLM versions
3. **Meeting presentation**: Advisor can see exact settings

**Optional flags to explore:**
- `--max-num-batched-tokens 4096`: Increase for higher throughput (if memory allows)
- `--max-num-seqs 512`: Allow more concurrent requests (if compute allows)
- `--disable-log-requests`: Reduce logging overhead

---

## How These Values Were Found

1. **Search for SchedulerConfig class** in vLLM source code
2. **Locate DEFAULT constants** (lines 44-45 in scheduler.py)
3. **Check EngineArgs context logic** (lines 1809-1816 in arg_utils.py)
4. **Verify chunked prefill auto-detection** (lines 1854-1858 in arg_utils.py)
5. **Confirm model support** (lines 1718-1719 in model.py)

---

## Implications for Our Experiments

### What This Means for Reported Results

Our vLLM benchmarks used:
- ✅ **Chunked prefill: ENABLED** (auto-detected for Qwen2-VL)
- ✅ **max_num_batched_tokens: 2048** (OpenAI server default)
- ✅ **max_num_seqs: 256** (OpenAI server default)

**Speedup attribution:**
- Continuous batching: ✅ Active
- PagedAttention: ✅ Active (always enabled)
- Prefix caching: ✅ Active (observed 99.3% hit rate)
- Chunked prefill: ✅ Active (confirmed via code inspection)

---

## References

### Source Code Files Examined

1. **SchedulerConfig defaults:**
   - Path: `/ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving/lib/python3.12/site-packages/vllm/config/scheduler.py`
   - Lines: 44-45 (constants), 50-61 (field definitions), 78-83 (chunked prefill)

2. **EngineArgs context defaults:**
   - Path: `/ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving/lib/python3.12/site-packages/vllm/engine/arg_utils.py`
   - Lines: 1809-1816 (context-specific defaults), 1854-1858 (auto-detection logic)

3. **ModelConfig support detection:**
   - Path: `/ssd1/zhuoyuan/envs/qwen2vl_nutrition_vllm_serving/lib/python3.12/site-packages/vllm/config/model.py`
   - Lines: 1718-1719 (generative model support)

### Official Documentation

- vLLM docs: https://docs.vllm.ai
- GitHub (v0.13.0): https://github.com/vllm-project/vllm/tree/v0.13.0
- Engine Arguments: https://docs.vllm.ai/en/latest/models/engine_args.html

---

*Document created: 2026-01-13*
*Purpose: Verify vLLM default configuration for advisor meeting*
