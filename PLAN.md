# Project Roadmap — AsrDemo (Qwen-Focused)

> Overview and development plan for the on-device AI Speech Assistant, now focused on Qwen3-ASR.

## Vision

A multi-phase on-device AI project: starting from high-precision ASR input, building towards a full loop of:
「Voice Input → On-device ASR → On-device LLM Intent Recognition → Tool Calling」.

## Technical Goals

| Goal | Description | Status |
|---|---|---|
| Qwen3-ASR Deployment | 0.6B LLM-based ASR on Android | ✅ Completed |
| Memory Management | mmap model loading for 1GB+ models | ✅ Completed |
| RTF Optimization | CPU-config tuning exhausted; warm device RTF ≈ 1.4–2.2 (length-dependent) | ⏸ CPU path capped |
| Agentic Loops | Integration with on-device LLM (Qwen-1.5B) | Planned |
| Tool Calling | Voice-controlled system actions | Planned |

## Phases

### Phase 1: High-Precision ASR (Current)

**✅ Completed:**
- Qwen3-ASR-0.6B cascade implementation (5 models).
- KV Cache management in Kotlin.
- Mel feature extraction (128-bin) parity with Python.
- Android model lifecycle management (Asset to Internal Storage).
- Memory-safe loading via mmap.

**⬜ Upcoming:**
- [ ] WER benchmark against Sherpa-Zipformer.
- [ ] Support for longer audio via KV Cache windowing.
- [ ] (Deferred) Per-stage timing instrumentation in `transcribe()` to characterize RTF-vs-length with real data.

### RTF Optimization Log

> All numbers are locally measured (本机实测), not projected. Recorded so dead ends aren't re-tried.

**Headline result:** warm on-device RTF ≈ **1.4–2.2** on a Snapdragon 888-class phone; PC reference RTF ≈ **0.386**.

**Key finding — RTF is utterance-length dependent, NOT a constant.** The same sentence reproduces the same RTF on repeat (this rules out warm/cold, DVFS, and thermal as the cause of the 1.4↔2.2 spread). Structural reason: Qwen3-ASR encodes audio as prompt tokens (~12.5 frames/s); the audio prefill is ≈ O(audio²), and every generated text token re-attends over all audio frames, while audio seconds grow only linearly → RTF rises with length. **A single RTF number is meaningless without stating utterance length.**

**Time breakdown (8 samples, prior measurement):** mel 2.7% / audio_frontend 3.3% / audio_transformer 4.7% / decoder prefill 11.2% / autoregressive gen 78.1% (decode total ≈ 89%); ≈ 118 ms/token, ≈ 11 tokens/utt. Decode dominates.

**Levers tried (CPU-config level — now exhausted):**

| Lever | Result |
|---|---|
| intra-op threads 2→4 | ✅ Effective (~28% RTF drop). Cap at big-core count (SD888 = 1 prime + 3 big). **Current setting.** |
| intra-op threads 4→6 | ❌ Harmful — spills onto A55 little cores; intra-op sync barrier gated by slowest thread → RTF jitter (1.8–2.6). |
| XNNPACK EP | ❌ No effect — decoder is INT8 **dynamic** quant (`MatMulInteger` + `DynamicQuantizeLinear`), unsupported by XNNPACK; the ~90% decode falls back to CPU. |
| QNN / HTP (NPU) | ⛔ Blocked — HTP requires **static QDQ** quant AND **fixed shapes**; our decoder is dynamic-quant + dynamic (growing KV) shapes; fp32 audio towers also can't run on HTP (HTP runs quantized only). Needs full fixed-shape + static-quant re-export. |
| INT4 `MatMulNBits` re-export | ❌ Slow even on PC/CPU (per-matmul 4-bit dequant overhead outweighs bandwidth saving). |
| FloatBuffer KV read (drop `.value` nested-array materialization; removed `flatten5D`) | ⚠️ RTF effect **inconclusive** (the before/after comparison was confounded by differing utterance lengths). Kept anyway — behavior-equivalent and cleaner. Needs same-sentence A/B to verdict. |

**Confirmed NOT a bug:** KV cache is already incremental (prefill once → single-token steps); no O(n²) recompute.

**Conclusion:** CPU-config optimization is exhausted. Warm device RTF ≈ 1.4–2.2 by utterance length is the realistic operating point on the current path. Getting below 1.0 requires **model-level** work — static QDQ re-export (also unblocks NPU), fixed-shape export for QNN/HTP, or an architecture change — most of which is currently blocked or unproven.

### Phase 2: Agentic Loops

- Integration with on-device LLM for Intent Recognition.
- Using ASR output directly as LLM prompt.
- Function Calling implementation.

## Project Structure

```
.
├── AGENT.md                  # Project spec and technical details (Qwen-Only)
├── PLAN.md                   # This roadmap
├── app/                      # Android Application module
└── models/                   # Model cache (Git ignored)
```

## Status Summary

- [x] Project architecture simplified to Qwen-only.
- [x] Memory and JNI conflict issues resolved.
- [x] RTF optimization explored — CPU-config path exhausted (see RTF Optimization Log).
- [ ] **Next Step**: Per-stage timing instrumentation to characterize RTF-vs-length; then decide whether model-level (static QDQ re-export) work is worth it.
