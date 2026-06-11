package com.example.asrdemo

import android.content.res.AssetManager
import com.k2fsa.sherpa.onnx.EndpointConfig
import com.k2fsa.sherpa.onnx.EndpointRule
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OfflinePunctuation
import com.k2fsa.sherpa.onnx.OfflinePunctuationConfig
import com.k2fsa.sherpa.onnx.OfflinePunctuationModelConfig
import com.k2fsa.sherpa.onnx.OnlineModelConfig
import com.k2fsa.sherpa.onnx.OnlineRecognizer
import com.k2fsa.sherpa.onnx.OnlineRecognizerConfig
import com.k2fsa.sherpa.onnx.OnlineStream
import com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig

/**
 * sherpa-onnx 流式识别引擎封装（中英双语 Zipformer transducer，int8，CPU 推理）。
 *
 * 模型: sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
 * 标点: sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12 (int8)
 * 均放置于 app/src/main/assets/ 下。
 *
 * 线程模型：所有方法须在同一个工作线程调用（与 AudioRecord 读取线程一致）。
 */
class SherpaAsrEngine(assetManager: AssetManager) {

    companion object {
        const val SAMPLE_RATE = 16000
        private const val MODEL_DIR = "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
        private const val PUNCT_MODEL = "punct-ct-transformer-zh-en/model.int8.onnx"
    }

    private val recognizer: OnlineRecognizer
    private var stream: OnlineStream
    private val punctuation: OfflinePunctuation

    init {
        val config = OnlineRecognizerConfig(
            featConfig = FeatureConfig(sampleRate = SAMPLE_RATE, featureDim = 80),
            modelConfig = OnlineModelConfig(
                transducer = OnlineTransducerModelConfig(
                    encoder = "$MODEL_DIR/encoder-epoch-99-avg-1.int8.onnx",
                    decoder = "$MODEL_DIR/decoder-epoch-99-avg-1.onnx",
                    joiner = "$MODEL_DIR/joiner-epoch-99-avg-1.int8.onnx",
                ),
                tokens = "$MODEL_DIR/tokens.txt",
                numThreads = 2,           // Snapdragon 888: 2 线程兼顾速度与发热
                provider = "cpu",
                modelType = "zipformer",
                debug = false,
            ),
            enableEndpoint = true,
            endpointConfig = EndpointConfig(
                rule1 = EndpointRule(false, 2.4f, 0.0f),  // 纯静音 2.4s 断句
                rule2 = EndpointRule(true, 1.0f, 0.0f),   // 说话后停顿 1.0s 断句（默认 1.4，略调短）
                rule3 = EndpointRule(false, 0.0f, 20.0f), // 单句最长 20s 强制断句
            ),
            decodingMethod = "greedy_search",
        )
        recognizer = OnlineRecognizer(assetManager, config)
        stream = recognizer.createStream()

        punctuation = OfflinePunctuation(
            assetManager,
            OfflinePunctuationConfig(
                model = OfflinePunctuationModelConfig(
                    ctTransformer = PUNCT_MODEL,
                    numThreads = 1,
                    debug = false,
                    provider = "cpu",
                ),
            ),
        )
    }

    /** 给最终识别结果添加中英文标点（CT-Transformer）。耗时约几十毫秒，仅对定稿句子调用。 */
    fun addPunctuation(text: String): String {
        if (text.isBlank()) return text
        return punctuation.addPunctuation(text)
    }

    /**
     * 送入一段音频样本（[-1, 1] 浮点），返回当前识别状态。
     *
     * @return Pair(当前文本, 是否到达端点)。到达端点时文本为该句最终结果，
     *         内部已自动 reset，可继续送音频识别下一句。
     */
    fun feed(samples: FloatArray): Pair<String, Boolean> {
        stream.acceptWaveform(samples, SAMPLE_RATE)
        while (recognizer.isReady(stream)) {
            recognizer.decode(stream)
        }
        val isEndpoint = recognizer.isEndpoint(stream)
        val text = recognizer.getResult(stream).text
        if (isEndpoint) {
            recognizer.reset(stream)
        }
        return Pair(text, isEndpoint)
    }

    /** 停止录音时调用：冲刷剩余音频，返回最后一句结果并重置。 */
    fun finish(): String {
        stream.inputFinished()
        while (recognizer.isReady(stream)) {
            recognizer.decode(stream)
        }
        val text = recognizer.getResult(stream).text
        recognizer.reset(stream)
        return text
    }

    /** 释放 native 资源。释放后本对象不可再用。 */
    fun release() {
        stream.release()
        recognizer.release()
        punctuation.release()
    }
}
