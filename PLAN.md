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
| RTF Optimization | INT8 CPU optimization for Snapdragon 888+ | In Progress |
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
- [ ] RTF optimization (aiming for < 0.3 on SD888).
- [ ] WER benchmark against Sherpa-Zipformer.
- [ ] Support for longer audio via KV Cache windowing.

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
- [ ] **Next Step**: Performance benchmarking and RTF optimization.
