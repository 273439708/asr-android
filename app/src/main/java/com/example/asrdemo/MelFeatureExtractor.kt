package com.example.asrdemo

import android.content.res.AssetManager
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

/**
 * 与 HuggingFace WhisperFeatureExtractor 数值对齐的 128-bin log-mel 提取器。
 *
 * 参数取自 Qwen3-ASR-0.6B/preprocessor_config.json:
 *   feature_extractor_type=WhisperFeatureExtractor, n_fft=400, hop_length=160, feature_size=128。
 *
 * 流程(逐步对齐 transformers spectrogram + WhisperFeatureExtractor._np_extract_fbank_features):
 *   reflect pad(n_fft/2=200) → periodic Hann(400) → 400 点 DFT → 功率谱(201 bin)
 *   → slaney mel 滤波(128×201, 由 asset mel_filters_128x201.bin 加载)
 *   → log10(clip(·,1e-10)) → max(·, globalMax-8) → (·+4)/4 → 丢弃最后一帧。
 *
 * 输出: 帧优先(frame-major) FloatArray, 长度 = T*128, 索引 = frame*128 + mel。
 *
 * 注: n_fft=400 非 2 的幂, 故用预计算三角表的直接 DFT(每帧 201×400)。短音频开销可接受;
 *     如需更快可改 Bluestein/混合基 FFT, 但需保持 400 点频率栅格不变(mel 滤波器按此构建)。
 */
class MelFeatureExtractor(assets: AssetManager) {

    private val nFft = 400
    private val hop = 160
    private val nMel = 128
    private val nFreq = nFft / 2 + 1   // 201
    private val pad = nFft / 2          // 200
    private val log10e = 0.4342944819032518f

    // periodic Hann: np.hanning(n_fft+1)[:-1] = 0.5 - 0.5*cos(2*pi*n/n_fft)
    private val window = FloatArray(nFft) { i ->
        (0.5 - 0.5 * cos(2.0 * PI * i / nFft)).toFloat()
    }

    private val melFilters: FloatArray  // [nMel*nFreq] mel-major(行=mel, 列=freq)
    private val cosT = FloatArray(nFreq * nFft)
    private val sinT = FloatArray(nFreq * nFft)

    init {
        val bytes = assets.open("qwen3-asr/mel_filters_128x201.bin").use { it.readBytes() }
        val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        melFilters = FloatArray(nMel * nFreq).also { fb.get(it) }

        for (k in 0 until nFreq) {
            val kb = k * nFft
            for (n in 0 until nFft) {
                val a = -2.0 * PI * k * n / nFft
                cosT[kb + n] = cos(a).toFloat()
                sinT[kb + n] = sin(a).toFloat()
            }
        }
    }

    fun compute(pcm: FloatArray): FloatArray {
        if (pcm.size <= pad) return FloatArray(0)

        // reflect 填充(与 np.pad(mode="reflect") 一致, 不重复边缘样本)
        val padded = FloatArray(pcm.size + 2 * pad)
        for (j in 0 until pad) padded[j] = pcm[pad - j]
        System.arraycopy(pcm, 0, padded, pad, pcm.size)
        for (k in 0 until pad) padded[pad + pcm.size + k] = pcm[pcm.size - 2 - k]

        val nFramesAll = 1 + (padded.size - nFft) / hop
        if (nFramesAll <= 1) return FloatArray(0)
        val nFrames = nFramesAll - 1   // 丢最后一帧, 与 Whisper 对齐

        val logMel = FloatArray(nFrames * nMel)
        var gmax = Float.NEGATIVE_INFINITY
        val fr = FloatArray(nFft)
        val power = FloatArray(nFreq)

        for (f in 0 until nFrames) {
            val base = f * hop
            for (n in 0 until nFft) fr[n] = padded[base + n] * window[n]

            for (k in 0 until nFreq) {
                var re = 0f
                var im = 0f
                val kb = k * nFft
                for (n in 0 until nFft) {
                    val s = fr[n]
                    re += s * cosT[kb + n]
                    im += s * sinT[kb + n]
                }
                power[k] = re * re + im * im
            }

            for (m in 0 until nMel) {
                var sum = 0f
                val mb = m * nFreq
                for (k in 0 until nFreq) sum += melFilters[mb + k] * power[k]
                val v = ln(max(1e-10f, sum)) * log10e
                logMel[f * nMel + m] = v
                if (v > gmax) gmax = v
            }
        }

        val floor = gmax - 8f
        for (i in logMel.indices) {
            val v = if (logMel[i] < floor) floor else logMel[i]
            logMel[i] = (v + 4f) / 4f
        }
        return logMel
    }
}
