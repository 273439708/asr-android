# AsrDemo — 端侧中英文语音识别 Demo

Android **离线**中英文语音识别演示。项目经历了两代实现:

- **初代(Zipformer 流式)**:sherpa-onnx + Zipformer transducer,真·流式、边说边出字,RTF < 0.2。
- **现行(Qwen3-ASR)**:Qwen3-ASR-0.6B,LLM 式整句识别,精度更高,RTF 1.4–2.2(随句长)。

当前主线代码为 **Qwen3-ASR**;Zipformer 初代实现保存在 git 历史(详见末尾「项目沿革与方案对比」)。全部计算在设备本地完成,不联网。

> 详细技术说明见 [`AGENT.md`](AGENT.md)(项目规格),开发路线与 RTF 优化日志见 [`PLAN.md`](PLAN.md)。

<p align="center">
  <img src="screenshot.png" alt="App screenshot" width="320">
</p>

## 特性(现行 Qwen3-ASR)

- **LLM 式 ASR**:Qwen3-ASR-0.6B,5 个 ONNX 模型级联 + Kotlin 手写 KV Cache 自回归贪心解码。
- **中英文混合**:依赖 Qwen3 tokenizer,原生支持中英文及混读。
- **纯端侧 / 离线**:ONNX Runtime CPU(INT8 动态量化 decoder),无需网络。
- **大模型内存安全**:模型经内部存储以 mmap 加载,避免 JVM Heap OOM;启用 `largeHeap`。
- **特征对齐**:`MelFeatureExtractor` 与 HuggingFace `WhisperFeatureExtractor` 数值对齐(128-bin,n_fft=400,hop=160,Slaney mel,reflect pad,periodic Hann,log10 归一)。

## ⚠️ 适用范围与诚实说明

- **整句离线,非流式**:当前是「点开始录音 → 点停止 → 对整段音频一次性识别」,不是边说边出字。(初代 Zipformer 才是流式,见末尾对比。)
- **面向高端机**:约需 **1.2 GB 运行内存**,目标设备为骁龙 888 级别(开发机:小米 11 Pro)。APK 仅打包 `arm64-v8a`。
- **性能(本机实测,非推算)**:热态设备 **RTF ≈ 1.4–2.2,随句子长度变化**(句子越长 RTF 越高);PC 参考 RTF ≈ 0.386。**RTF > 1 表示比实时慢**,即 10s 音频需要 14–22s 处理。RTF 不是常数,脱离句长谈单个数字没有意义——详见 [`PLAN.md` 的 RTF Optimization Log](PLAN.md)。

## 技术栈(版本已锁定)

| 组件 | 版本 |
|---|---|
| Gradle | 9.4.1 |
| AGP | 9.2.1 |
| Kotlin | 2.2.10 |
| Compose BOM | 2026.02.01 |
| onnxruntime-android | 1.20.0 |
| compileSdk / targetSdk | 36 / 36 |
| minSdk | 24 |
| ABI | arm64-v8a |

## 代码结构

```
app/src/main/java/com/example/asrdemo/
├── MainActivity.kt         # Compose UI + AudioRecord(16kHz 单声道)录音控制,显示 RTF
├── QwenAsrEngine.kt        # Qwen3-ASR 引擎:5 模型级联 + KV Cache 解码循环
├── MelFeatureExtractor.kt  # 128-bin log-mel 提取(与 WhisperFeatureExtractor 对齐)
└── TextPostProcessor.kt    # 英文大小写、中英文边界空格、标点润色
```

### 推理数据流

```
AudioRecord(16kHz) → 累积整段音频 → transcribe()
  → MelFeatureExtractor(128-bin log-mel)
  → runFrontend(Conv 下采样)  → runTransformer(分窗注意力 Encoder)
  → runDecodeLoop(LLM Decoder + KV Cache 贪心解码) → TextPostProcessor → UI
```

## 模型文件(需自行准备)

模型与词表**不纳入 Git**(见 `.gitignore`),需放到 `app/src/main/assets/qwen3-asr/`:

| 文件 | 说明 |
|---|---|
| `audio_frontend.onnx` | 卷积前端 |
| `audio_transformer.onnx` | 音频 Encoder |
| `embed.onnx` | Token embedding |
| `decoder.onnx` | 28 层 LLM Decoder(INT8 动态量化) |
| `lm_head.onnx` | 词表投影 |
| `tokens.txt` | 词表(由 `vocab.json` 转换) |
| `mel_filters_128x201.bin` | Slaney mel 滤波器组(float32,mel-major [128,201]) |

### 辅助脚本

```bash
# 从 vocab.json 生成 tokens.txt
python convert_vocab.py

# 从 Qwen3-ASR-0.6B/preprocessor_config.json 导出 mel 滤波器组
python export_mel_filters.py
```

> 脚本内的源路径(如 `M:/models/Qwen3-ASR-0.6B`)按本地环境硬编码,使用前请按需修改。

## 构建与部署

```bash
# 构建 Debug APK
./gradlew :app:assembleDebug

# 安装并启动
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.example.asrdemo/.MainActivity
```

首次启动会把 assets 中的模型拷贝到内部存储(约 0.9 GB),加载需稍候;应用需要 `RECORD_AUDIO` 权限。

## 项目沿革与方案对比

项目从一个 **Zipformer 流式方案** 起步,后迁移到 **Qwen3-ASR LLM 方案**。两者不是替代关系,而是**速度/流式 ↔ 精度**的取舍,各有适用场景:

| 维度 | 初代 · Zipformer(git 历史) | 现行 · Qwen3-ASR(主线) |
|---|---|---|
| 框架 | sherpa-onnx AAR v1.13.2 | 手写 ONNX Runtime 级联 |
| 模型 | streaming-zipformer-bilingual-zh-en-2023-02-20 | Qwen3-ASR-0.6B(5 模型级联) |
| 架构 | Zipformer **transducer**(encoder/decoder/joiner) | LLM 自回归 + KV Cache |
| 量化 | int8(encoder/joiner) | INT8 动态量化 decoder |
| 模式 | **流式**(边说边出字) | 整句离线(录完再识别) |
| **RTF** | **< 0.2**(比实时快 5 倍+) | **1.4–2.2**(随句长,比实时慢) |
| 内存 | 较小 | 约 1.2 GB |
| 线程 | numThreads=2(识别) | intra-op 4 |
| 标点 | CT-Transformer 模型(int8) | 规则后处理 |
| 端点检测 | 静音 2.4s / 停顿 1.0s / 最长 20s | 无(手动停止) |
| 取舍 | 快、流式、轻量 | 精度/鲁棒性更高,但慢且离线 |

**初代 Zipformer 方案要点(供复现/对照)**:

- ASR 模型:`sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20`(mobile 版,`encoder/joiner` int8 + `decoder` fp)。
- 标点模型:`sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12`(int8),仅对定稿句调用以省 RTF。
- 数据流:`AudioRecord(100ms/帧)` → `acceptWaveform`/`decode` → partial 实时刷新,端点触发定稿入列表。
- RTF 显示:累计推理耗时 / 累计音频时长,每 500ms 刷新。
- 封装类 `SherpaAsrEngine`(`OnlineRecognizer` + `OfflinePunctuation`)可在历史提交中找到。

> `PLAN.md` 中 “WER benchmark against Sherpa-Zipformer” 一项即用于量化二者精度差(Qwen 预期更高)。

## 路线图

- **Phase 1 — 高精度端侧 ASR**:✅ Qwen3-ASR-0.6B 上设备、INT8 级联、Kotlin KV Cache、Mel 对齐已完成。
- **Phase 2 — Agentic Loops**:⬜ 端侧 LLM 意图识别、Function Calling、端到端语音助手闭环。

详见 [`PLAN.md`](PLAN.md)。
