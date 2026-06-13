# AGENT.md — AsrDemo Project Specification (Qwen-Only)

> Android on-device bilingual (Chinese/English) real-time ASR Demo.
> Powered by **Qwen3-ASR-0.6B** (Offline/LLM-based).
> This document is for AI Agents / developers to quickly understand the project.
> For the overall project roadmap, see `PLAN.md`.

## Project Overview

- **Goal**: High-precision on-device ASR on high-performance Android devices (e.g., Snapdragon 888).
- **Package**: `com.example.asrdemo`
- **Inference**: 
    - **Qwen3-ASR-0.6B**: High precision, CPU int8 (级联推理), requires ~1.2GB RAM.

## Tech Stack (Versions are pinned)

 Component | Version | Description |
---|---|---|
 Gradle | 9.4.1 | |
 AGP | 9.2.1 | |
 Kotlin | 2.2.10 | |
 Compose BOM | 2026.02.01 | |
 onnxruntime-android | 1.20.0 | Official ONNX Runtime for Android |
 compileSdk | 36.1 | |

## Commands (Build & Deploy)

```bash
# Build
./gradlew :app:assembleDebug

# Install & Launch
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.example.asrdemo/.MainActivity
```

## Code Structure

```
app/src/main/java/com/example/asrdemo/
├── MainActivity.kt         # Compose UI + AudioRecord Controller
├── QwenAsrEngine.kt        # Custom Qwen3-ASR Engine (5-model cascade + KV Cache)
├── MelFeatureExtractor.kt   # 128-bin Mel spectrogram implementation for Qwen3
└── TextPostProcessor.kt    # Casing, Spacing, and Punctuation polishing
```

### Data Flow

#### Qwen3-ASR (Offline / Full Sentence)
```
AudioRecord -> Accumulate Audio -> transcribe() 
  -> MelFeatureExtractor (128-bin)
  -> runFrontend (Conv) -> runTransformer (Encoder)
  -> runDecodeLoop (LLM Decoder + KV Cache loop) -> UI Update
```

### Key Design & Fixes

- **Memory Optimization**: Models are loaded via file paths (mmap) to avoid JVM Heap OOM.
- **Large Memory Support**: `android:largeHeap="true"` enabled for 0.6B model support.
- **Bilingual Support**: Native support for Chinese and English via Qwen3 tokenizer.

## Model Files

Models are excluded from Git. 

### Qwen3-ASR-0.6B
- **Files**: `audio_frontend.onnx`, `audio_transformer.onnx`, `embed.onnx`, `decoder.onnx`, `lm_head.onnx`.
- **Vocab**: `tokens.txt` (converted from `vocab.json`).
- **Location**: `app/src/main/assets/qwen3-asr/`

## Progress

### ✅ Phase 1: High-Precision LLM-ASR
- Qwen3-ASR-0.6B integration on Android.
- INT8 Model cascade implementation.
- KV Cache manual management in Kotlin.
- Mel feature extraction parity.
- **Status**: Android implementation completed.

### ⬜ Phase 2: Agentic Loops
- [ ] Integration with on-device LLM for Intent Recognition.
- [ ] Function Calling implementation.
- [ ] End-to-end Voice Assistant loop.
