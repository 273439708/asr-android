# Project Roadmap — AsrDemo

> Overview and development plan for the on-device AI Speech Assistant.

## Vision

A multi-phase on-device AI project: starting from ASR input, building towards a full loop of:
「Voice Input → On-device ASR → On-device LLM Intent Recognition → Tool Calling」.

## Technical Goals

| Goal | Description | Status |
|---|---|---|
| On-device Deployment | Android integration with ONNX Runtime | ✅ Completed |
| Real-time Inference | Performance optimization (RTF < 0.2) | ✅ Completed |
| Punctuation Recovery | Post-processing for readability | ✅ Completed |
| Model Conversion | PyTorch to ONNX export + operator compatibility | In Progress |
| Quantization | INT8/INT4 quantization and accuracy validation | Planned |
| Benchmarking | Automated scripts for WER / RTF / Memory | Planned |
| LLM Integration | Small LLM (Qwen/Llama) deployment on-device | Planned |

## Phases

### Phase 1: ASR Deployment (Current)

**✅ Completed:**
- sherpa-onnx integration with Zipformer transducer.
- Full microphone-to-text pipeline (16kHz AudioRecord).
- Text post-processing (Casing, Spacing, Half-width conversion).
- Punctuation recovery (CT-Transformer).
- Real-time performance metrics (RTF tracking).

**⬜ Upcoming:**
- [ ] PyTorch to ONNX export pipeline for custom models.
- [ ] Manual INT8 quantization and accuracy (WER) verification.
- [ ] Systematic benchmark reports (WER / RTF / Latency / Memory).
- [ ] Hotword boosting support.

### Phase 2: On-device LLM

- Integration of small language models (e.g., Qwen-0.5B/1.8B).
- Inference via llama.cpp (k-quants) or ONNX Runtime GenAI.
- Metrics: First token latency, Token/s, KV Cache optimization.

### Phase 3: Agentic Loops

- Voice → ASR → LLM → Function Calling (Simulated tools like Navigation/Media).
- End-to-end latency optimization for natural interaction.

## Project Structure

```
.
├── AGENT.md                  # Project spec and technical details
├── PLAN.md                   # This roadmap
├── app/                      # Android Application module
└── models/                   # Model cache (Git ignored)
```

## Status Summary

- [x] Project initialization and environment setup.
- [x] Phase 1 Android Demo verified on real devices.
- [ ] **Next Step**: Custom model export and quantization pipeline.
- [ ] Phase 1 Benchmark Report.
- [ ] Phase 2 LLM Integration.
