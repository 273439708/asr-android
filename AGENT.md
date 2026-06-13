# AGENT.md — AsrDemo Project Specification

> Android on-device bilingual (Chinese/English) real-time ASR Demo (sherpa-onnx + Zipformer + Jetpack Compose).
> This document is for AI Agents / developers to quickly understand the project.
> For the overall project roadmap, see `PLAN.md`.

## Project Overview

- **Goal**: Real-time on-device streaming ASR on high-performance Android devices (e.g., Snapdragon 888, arm64-v8a).
- **Package**: `com.example.asrdemo`
- **Inference**: CPU int8 (Optimized for mobile CPU inference).

## Tech Stack (Versions are pinned)

| Component | Version | Description |
|---|---|---|
| Gradle | 9.4.1 | |
| AGP | 9.2.1 | Built-in Kotlin, do not apply `kotlin-android` plugin separately |
| Kotlin | 2.2.10 | `kotlin-compose` plugin |
| Compose BOM | 2026.02.01 | |
| JDK | JBR 21 | Recommended for consistency |
| sherpa-onnx | v1.13.2 | Official AAR in `app/libs/` |
| compileSdk | 36.1 | |
| minSdk / targetSdk | 24 / 36 | |

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
├── MainActivity.kt        # Compose UI + AudioRecord + Main Loop
├── SherpaAsrEngine.kt     # sherpa-onnx Wrapper: OnlineRecognizer + OfflinePunctuation
└── TextPostProcessor.kt   # Text processing: Casing, Spacing, Punctuation conversion
```

### Data Flow

```
AudioRecord (16kHz mono PCM16)
  → ShortArray → FloatArray
  → SherpaAsrEngine.feed() → (text, isEndpoint)
  → TextPostProcessor.process()
  → [Final only] addPunctuation() → polishPunctuation()
  → UI Update
```

### Key Design

- Async model loading.
- Endpoint rules: 1.0s silence for sentence breaking.
- Punctuation recovery is only applied to final results (to save CPU/RTF).
- Real-time RTF tracking.
- `FLAG_KEEP_SCREEN_ON` during recognition.

## Model Files

Models are excluded from Git. Fetch them via:

```bash
# 1) sherpa-onnx AAR -> app/libs/
curl -L -o app/libs/sherpa-onnx-1.13.2.aar \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.2/sherpa-onnx-1.13.2.aar

# 2) ASR Model (Mobile version) -> assets
# Download sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-mobile.tar.bz2
# Extract: encoder-epoch-99-avg-1.int8.onnx, decoder-epoch-99-avg-1.onnx, joiner-epoch-99-avg-1.int8.onnx, tokens.txt
# Place in: app/src/main/assets/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/

# 3) Punctuation Model -> assets
# Download sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8.tar.bz2
# Extract: model.int8.onnx
# Place in: app/src/main/assets/punct-ct-transformer-zh-en/
```

## Progress

### ✅ Phase 1: Basic Recognition
- Project setup, AAR integration, Zipformer streaming, Microphone capture, Compose UI.
- Verified on real devices.

### ✅ Phase 2: UX Optimization
- Text post-processing (Casing/Spacing).
- Real-time RTF display.
- Punctuation recovery (CT-Transformer).

### ⬜ Todo / Roadmap
- [ ] Hotword boosting (modified_beam_search).
- [ ] Partial text punctuation (throttled).
- [ ] Automated benchmark scripts.
