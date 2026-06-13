package com.example.asrdemo

import android.content.Context
import android.content.res.AssetManager
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.Collections
import kotlin.math.*

/**
 * Qwen3-ASR-0.6B 推理引擎。
 */
class QwenAsrEngine(private val context: Context) {

    private val assetManager: AssetManager = context.assets
    private val env = OrtEnvironment.getEnvironment()
    private val sessions: Map<String, OrtSession>
    private val tokens: List<String>
    private val melExtractor = MelFeatureExtractor(context.assets)

    private val numLayers = 28
    private val numHeads = 8
    private val headDim = 128
    private val numMelBins = 128
    private val chunkFrames = 100
    
    init {
        val opts = OrtSession.SessionOptions()
        // 解码占 ~90% 耗时，受 decoder 矩阵乘的 intra-op 并行度影响最大。
        // 线程数取在线核数(上限 4)：SD888 等 1+3+4 架构上≈大核数，避免调度到小核反而拖慢。
        // 注：最优值需真机实测，可按设备微调此上限。
        val threads = Runtime.getRuntime().availableProcessors().coerceIn(2, 4)
        opts.setIntraOpNumThreads(threads)

        // 为避免 JVM Heap OOM (910MB 模型)，将 Asset 拷贝到内部存储，使用文件路径加载（Mmap）
        val modelDir = File(context.filesDir, "qwen3-asr")
        if (!modelDir.exists()) modelDir.mkdirs()

        val modelFiles = listOf(
            "audio_frontend.onnx",
            "audio_transformer.onnx",
            "embed.onnx",
            "decoder.onnx",
            "lm_head.onnx"
        )
        
        val sessionMap = mutableMapOf<String, OrtSession>()
        for (name in modelFiles) {
            val file = File(modelDir, name)
            if (!file.exists() || file.length() == 0L) {
                copyAssetToFile("qwen3-asr/$name", file)
            }
            val key = name.removeSuffix(".onnx").removePrefix("audio_")
            sessionMap[key] = env.createSession(file.absolutePath, opts)
        }
        sessions = sessionMap
        
        // 显式指定 UTF-8 编码读取 tokens.txt，防止在部分 Android 系统上出现乱码
        tokens = assetManager.open("qwen3-asr/tokens.txt").bufferedReader(Charsets.UTF_8).readLines()
    }

    private fun copyAssetToFile(assetPath: String, outFile: File) {
        assetManager.open(assetPath).use { input ->
            FileOutputStream(outFile).use { output ->
                input.copyTo(output)
            }
        }
    }

    fun transcribe(pcm: FloatArray): String {
        // mel: 帧优先(frame-major) FloatArray, 长度 = T*128, 索引 = frame*128 + mel
        val mel = computeMel(pcm)
        val t = mel.size / numMelBins            // mel 帧数 T
        if (t == 0) return ""

        val nFull = t / chunkFrames
        val tail = t % chunkFrames
        val n = nFull + if (tail > 0) 1 else 0   // 含尾块(补零)

        // 构建 audio_frontend 输入 [n,1,128,100]:
        //   ONNX 期望「通道内 mel 优先(mel×frame)」，而 mel 是 frame×mel，需逐元素转置。
        //   尾块超出 T 的帧补零(chunkBuf 默认 0)。等价参考实现:
        //   padded[:T]=mel.T; chunks=padded.reshape(n,100,128).transpose(0,2,1)
        val chunkBuf = FloatArray(n * numMelBins * chunkFrames)
        for (nn in 0 until n) {
            for (b in 0 until chunkFrames) {
                val frame = nn * chunkFrames + b
                if (frame >= t) continue
                val melBase = frame * numMelBins
                for (a in 0 until numMelBins) {
                    chunkBuf[((nn * numMelBins) + a) * chunkFrames + b] = mel[melBase + a]
                }
            }
        }
        val frontendOut = runFrontend(chunkBuf, n)   // [n][13][896]

        // 每块有效帧: 满块 13, 尾块 = convOutLen(tail)(3 层 stride-2 卷积后的有效帧数)
        val valid = IntArray(n) { 13 }
        if (tail > 0) valid[n - 1] = convOutLen(tail)
        val s = valid.sum()
        if (s == 0) return ""

        // 拼接有效帧 → hidden [s,896]
        val hidden = FloatArray(s * 896)
        var row = 0
        for (i in 0 until n) {
            for (j in 0 until valid[i]) {
                System.arraycopy(frontendOut[i][j], 0, hidden, row * 896, 896)
                row++
            }
        }

        val audioEmbeds = runTransformer(hidden, s)
        return runDecodeLoop(audioEmbeds, s)
    }

    /** 单块 t(<=100) 帧 mel 经 3 层 stride-2 conv 的有效输出帧数(逐级 ceil/2)。 */
    private fun convOutLen(tIn: Int): Int {
        var x = tIn
        repeat(3) { x = (x - 1) / 2 + 1 }
        return x
    }

    private fun runFrontend(chunkBuf: FloatArray, n: Int): Array<Array<FloatArray>> {
        val tensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(chunkBuf),
            longArrayOf(n.toLong(), 1, numMelBins.toLong(), chunkFrames.toLong())
        )
        val inputs = Collections.singletonMap("chunks", tensor)
        val session = sessions["frontend"]!!
        val res = session.run(inputs)
        @Suppress("UNCHECKED_CAST")
        val out = res.get(0).value as Array<Array<FloatArray>>   // [n,13,896]
        res.close()
        tensor.close()
        return out
    }

    private fun runTransformer(hidden: FloatArray, s: Int): FloatArray {
        val hiddenTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(hidden), longArrayOf(s.toLong(), 896))
        // 分窗注意力 mask（与参考实现一致）：8s/104 帧一窗的 block-diagonal，
        // 窗内 0、窗外大负值。全 0 mask 仅对单窗(≤8s)音频等价，长音频会跨窗串扰 → 乱码。
        val window = 104
        val neg = -1e9f
        val mask = FloatArray(s * s) { neg }
        var start = 0
        while (start < s) {
            val end = minOf(start + window, s)
            for (r in start until end) {
                val rowBase = r * s
                for (c in start until end) mask[rowBase + c] = 0f
            }
            start = end
        }
        val maskTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(mask), longArrayOf(1, 1, s.toLong(), s.toLong()))
        
        val inputs = mapOf("hidden" to hiddenTensor, "attn_mask" to maskTensor)
        val session = sessions["transformer"]!!
        val res = session.run(inputs)
        val out = res.get(0).value as Array<FloatArray>
        res.close()
        val flat = FloatArray(s * 1024)
        var idx = 0
        for (i in 0 until s) for (j in 0 until 1024) flat[idx++] = out[i][j]
        return flat
    }

    private fun runDecodeLoop(audioEmbeds: FloatArray, s: Int): String {
        // Qwen3-ASR 官方 chat 模板（与 PyTorch 贪心序列逐 token 对齐的参考实现一致）：
        //   <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
        //   [audio]<|audio_end|><|im_end|>\n<|im_start|>assistant\n
        // 关键：role token 必须是 system=8948 / user=872 / assistant=77091；
        // 之前误用 1183（其实是字符串 "[\""）当 user，且漏了 system 轮、audio_end 后多了换行，
        // 模板偏离训练分布 → decoder 生成乱码。
        val imStartId = 151644L      // <|im_start|>
        val imEndId = 151645L        // <|im_end|>
        val systemId = 8948L         // "system"
        val userId = 872L            // "user"
        val assistantId = 77091L     // "assistant"
        val audioStartId = 151669L   // <|audio_start|>
        val audioEndId = 151670L     // <|audio_end|>
        val audioPadId = 151671L     // <|audio_pad|>（占位，embedding 会被音频向量覆盖）
        val nlId = 198L              // "\n"

        val promptIds = mutableListOf<Long>()
        // system 轮（内容为空）
        promptIds.add(imStartId); promptIds.add(systemId); promptIds.add(nlId)
        promptIds.add(imEndId); promptIds.add(nlId)
        // user 轮 + 音频
        promptIds.add(imStartId); promptIds.add(userId); promptIds.add(nlId)
        promptIds.add(audioStartId)
        val audioPadStartIdx = promptIds.size
        for (i in 0 until s) promptIds.add(audioPadId)
        promptIds.add(audioEndId)
        promptIds.add(imEndId); promptIds.add(nlId)
        // assistant 轮（待生成）
        promptIds.add(imStartId); promptIds.add(assistantId); promptIds.add(nlId)
        
        val inputIdsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(promptIds.toLongArray()), longArrayOf(1, promptIds.size.toLong()))
        val sessionEmbed = sessions["embed"]!!
        val embedRes = sessionEmbed.run(Collections.singletonMap("input_ids", inputIdsTensor))
        val embeds = embedRes.get(0).value as Array<Array<FloatArray>>
        embedRes.close()
        
        // 将音频 Embedding 插入到 <|audio_pad|> 位置
        for (i in 0 until s) {
            val audioVec = audioEmbeds.sliceArray(i * 1024 until (i + 1) * 1024)
            embeds[0][audioPadStartIdx + i] = audioVec
        }

        var pastK = FloatArray(0)
        var pastV = FloatArray(0)
        var p = 0
        
        val resultTokens = mutableListOf<Long>()
        var currentEmbeds = embeds
        
        val sessionDecoder = sessions["decoder"]!!
        val sessionLmHead = sessions["lm_head"]!!

        for (step in 0 until 200) {
            val l = currentEmbeds[0].size
            val embedsTensor = OnnxTensor.createTensor(env, flatten3D(currentEmbeds), longArrayOf(1, l.toLong(), 1024))
            val pkTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(pastK), longArrayOf(numLayers.toLong(), 1, numHeads.toLong(), p.toLong(), headDim.toLong()))
            val pvTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(pastV), longArrayOf(numLayers.toLong(), 1, numHeads.toLong(), p.toLong(), headDim.toLong()))
            
            val decInputs = mapOf("inputs_embeds" to embedsTensor, "past_k" to pkTensor, "past_v" to pvTensor)
            val decRes = sessionDecoder.run(decInputs)
            
            val lastHidden = decRes.get(0).value as Array<Array<FloatArray>>
            val lastVec = lastHidden[0][l - 1]
            
            pastK = flatten5D(decRes.get(1).value as Array<Array<Array<Array<FloatArray>>>>)
            pastV = flatten5D(decRes.get(2).value as Array<Array<Array<Array<FloatArray>>>>)
            p += l
            decRes.close()
            
            val hTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(lastVec), longArrayOf(1, 1, 1024))
            val logitsRes = sessionLmHead.run(Collections.singletonMap("hidden", hTensor))
            val logits = (logitsRes.get(0).value as Array<Array<FloatArray>>)[0][0]
            logitsRes.close()
            
            val nextToken = logits.indices.maxByOrNull { logits[it] }!!.toLong()
            
            // 遇到结束符停止
            if (nextToken == 151643L || nextToken == 151645L) break
            resultTokens.add(nextToken)
            
            val nextIdsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(nextToken)), longArrayOf(1, 1))
            val nextEmbedRes = sessionEmbed.run(Collections.singletonMap("input_ids", nextIdsTensor))
            currentEmbeds = nextEmbedRes.get(0).value as Array<Array<FloatArray>>
            nextEmbedRes.close()
        }
        
        return decodeTokens(resultTokens)
    }

    /**
     * 实现 Qwen/GPT-2 风格的 Byte-level BPE 解码。
     * 过滤掉特殊 Token 和 <unk>。
     */
    private fun decodeTokens(tokenIds: List<Long>): String {
        if (tokenIds.isEmpty()) return ""

        // Qwen3-ASR 的原始输出是结构化序列：
        //   [语种标签 tokens] <asr_text>(151704) [转写文本 tokens]
        // 例如 "language Chinese<asr_text>甚至出现…"。只保留 <asr_text> 之后的转写文本，
        // 否则会把 "language Chinese"/"language none" 等语种前缀当成识别结果输出。
        val asrTextId = 151704L
        val idx = tokenIds.indexOf(asrTextId)
        val textIds = if (idx >= 0) tokenIds.subList(idx + 1, tokenIds.size) else tokenIds

        val byteList = mutableListOf<Byte>()
        for (id in textIds) {
            val token = tokens.getOrElse(id.toInt()) { "" }
            // 跳过特殊 Token（如 <|...|>）和 <unk>
            if (token.startsWith("<|") || token == "<unk>") continue
            
            for (char in token) {
                byteList.add(qwenByteEncoder[char] ?: char.code.toByte())
            }
        }
        
        return String(byteList.toByteArray(), Charsets.UTF_8).trim()
    }

    // Qwen 字节映射表 (Unicode 到 原始字节)
    private val qwenByteEncoder: Map<Char, Byte> by lazy {
        val map = mutableMapOf<Char, Byte>()
        fun range(start: Int, end: Int) = start..end
        val n = (range(33, 126) + range(161, 172) + range(174, 255)).toList()
        var i = 0
        for (b in 0..255) {
            if (b in n) {
                map[b.toChar()] = b.toByte()
            } else {
                map[(256 + i).toChar()] = b.toByte()
                i++
            }
        }
        map
    }

    private fun flatten3D(arr: Array<Array<FloatArray>>): FloatBuffer {
        val b = FloatBuffer.allocate(arr.size * arr[0].size * arr[0][0].size)
        for (i in arr.indices) for (j in arr[i].indices) b.put(arr[i][j])
        b.flip()
        return b
    }
    
    private fun flatten5D(arr: Array<Array<Array<Array<FloatArray>>>>): FloatArray {
        val flat = FloatArray(arr.size * arr[0].size * arr[0][0].size * arr[0][0][0].size * arr[0][0][0][0].size)
        var idx = 0
        for (i in arr.indices) for (j in arr[i].indices) for (k in arr[i][j].indices) 
            for (l in arr[i][j][k].indices) for (m in arr[i][j][k][l].indices) flat[idx++] = arr[i][j][k][l][m]
        return flat
    }

    private fun computeMel(pcm: FloatArray): FloatArray {
        return melExtractor.compute(pcm)
    }

    fun release() {
        sessions.values.forEach { it.close() }
        env.close()
    }
}
