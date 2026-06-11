# AGENT.md — AsrDemo 项目说明

> Android 端侧中英文混合实时语音识别 Demo（sherpa-onnx + Zipformer + Jetpack Compose）。
> 本文件供 AI Agent / 开发者快速接手项目使用。

## 项目概况

- **目标**：在小米 11 Pro（Snapdragon 888, arm64-v8a, Android 14/API 34）上做端侧实时中英文混合流式语音识别
- **路径**：`M:\projects\AsrDemo`
- **包名**：`com.example.asrdemo`
- **推理**：CPU int8（SD888 无可用 NPU 预编译支持，QNN 预编译 context 只支持 SM8850）

## 技术栈与版本（重要，勿随意升级）

| 组件 | 版本 | 说明 |
|---|---|---|
| Gradle | 9.4.1 | wrapper 从 M:/android-project 复制，命中本地缓存 |
| AGP | 9.2.1 | **内置 Kotlin**，不能再 apply `kotlin-android` 插件 |
| Kotlin | 2.2.10 | 仅 `kotlin-compose` 插件 |
| Compose BOM | 2026.02.01 | |
| JDK | JBR 21 | `gradle.properties` 中 `org.gradle.java.home=M:/Android/jbr`，避免联网下载 toolchain |
| sherpa-onnx | v1.13.2 | 官方 AAR 放在 `app/libs/`（JitPack 已失效 404） |
| compileSdk | 36.1 | `compileSdk { version = release(36) { minorApiLevel = 1 } }` |
| minSdk / targetSdk | 24 / 36 | |

## 常用命令（Git Bash 环境）

```bash
# 构建（必须先 cd 到项目目录，否则 gradle 在错误目录找 settings）
cd /m/projects/AsrDemo && cmd //c "M:\projects\AsrDemo\gradlew.bat :app:assembleDebug --console=plain"

# 安装 / 启动 / 截图
"M:/android-sdk/platform-tools/adb.exe" install -r app/build/outputs/apk/debug/app-debug.apk
"M:/android-sdk/platform-tools/adb.exe" shell am start -n com.example.asrdemo/.MainActivity
"M:/android-sdk/platform-tools/adb.exe" exec-out screencap -p > /tmp/shot.png
```

注意：真机走无线 ADB，闲置会掉线（`no devices found` 时请用户重新连接）。

## 代码结构

```
app/src/main/java/com/example/asrdemo/
├── MainActivity.kt        # Compose UI + AudioRecord 采集 + 工作线程驱动识别
├── SherpaAsrEngine.kt     # sherpa-onnx 封装：OnlineRecognizer(ASR) + OfflinePunctuation(标点)
└── TextPostProcessor.kt   # 文本后处理：大小写/中英空格/全角标点转半角
```

### 数据流

```
AudioRecord(VOICE_RECOGNITION, 16kHz mono PCM16, 100ms buffer)
  → ShortArray → FloatArray(/32768f)
  → SherpaAsrEngine.feed() → (text, isEndpoint)
  → TextPostProcessor.process()                     # partial 与 final 都做
  → [仅 final] addPunctuation() → polishPunctuation()  # 标点恢复 + 全角转半角
  → runOnUiThread 更新 Compose 状态
```

### 关键设计

- 模型加载放后台线程（`asr-init`），加载完才显示"开始识别"按钮
- endpoint 规则：rule2 停顿 1.0s 断句（默认 1.4s 调短）、rule3 单句最长 20s
- 标点**只对定稿句子做**（partial 每 100ms 刷新，跑标点模型会抬高 RTF/发热）
- RTF 统计：累计推理 nanoTime / 累计音频时长，每 500ms 刷新到 UI 副标题
- 识别期间 `FLAG_KEEP_SCREEN_ON` 保持常亮

## 模型文件（不入 git，需手动获取）

git 排除了大模型与 AAR（encoder 105MB 超 GitHub 单文件上限）。恢复方法：

```bash
cd /m/projects/AsrDemo
# 1) sherpa-onnx AAR → app/libs/
curl -L -o app/libs/sherpa-onnx-1.13.2.aar \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.2/sherpa-onnx-1.13.2.aar

# 2) ASR 模型（mobile 版）→ assets
curl -L -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-mobile.tar.bz2
# 解压后将 encoder int8 / decoder fp32 / joiner int8 / tokens.txt 4 个文件放入
# app/src/main/assets/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/

# 3) 标点模型 → assets
curl -L -O https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8.tar.bz2
# 解压后将 model.int8.onnx 放入 app/src/main/assets/punct-ct-transformer-zh-en/
```

assets 最终布局：

```
app/src/main/assets/
├── sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/
│   ├── encoder-epoch-99-avg-1.int8.onnx   (105MB)
│   ├── decoder-epoch-99-avg-1.onnx        (14MB, fp32 精度更好)
│   ├── joiner-epoch-99-avg-1.int8.onnx    (3MB)
│   └── tokens.txt
└── punct-ct-transformer-zh-en/
    └── model.int8.onnx                    (75MB)
```

## 进度

### ✅ 阶段 1：基础识别（已完成并真机验证）
- 项目搭建、AAR 集成、双语 Zipformer 流式识别、麦克风采集、Compose UI
- 真机实测中英文识别与 endpoint 断句均正常，进程 ~328MB PSS

### ✅ 阶段 2：体验优化（已完成并真机验证）
- 英文大小写规范化 + 中英边界空格（TextPostProcessor）
- 实时 RTF 显示（实测 0.15~0.19，余量充足）
- 识别时屏幕常亮
- **标点恢复**：CT-Transformer zh-en int8，中文全角/英文转半角，问号判断准确

### ⬜ 待办 / 候选
- [ ] 热词增强：需换 `modified_beam_search` 解码 + bpe 词表文件（bpe.vocab），v1.13.2 AAR API 已支持 `createStream(hotwords)`
- [ ] partial 文本也加标点（500ms 节流方案，RTF 预估升至 ~0.3）
- [ ] 清理 `models/` 下载缓存（~700MB，tarball + 解压目录，可直接删）
- [ ] 发布 release 包（当前 release buildType 关闭了 optimization）

## 已踩过的坑（勿重复）

1. **AGP 9.x 内置 Kotlin**：apply `kotlin-android` 会报 "Cannot add extension with name 'kotlin'"
2. **JitPack 失效**：`com.github.k2fsa:sherpa-onnx` 404，必须用 GitHub Releases AAR 本地引用
3. **toolchain 联网失败**：删 `gradle-daemon-jvm.properties`，用 `org.gradle.java.home` 指 JBR
4. **gradlew 必须在项目目录下执行**（cmd //c 不继承 bash 的 cwd 语义时会在错误目录找 build）
5. **构建失败但 adb install 仍 Success**：装的是旧 APK，务必先确认 BUILD SUCCESSFUL
6. **aapt 压缩**：`androidResources { noCompress += listOf("onnx", "txt") }` 已配置，避免安装时解压开销
