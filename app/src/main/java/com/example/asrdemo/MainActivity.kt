package com.example.asrdemo

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import kotlin.concurrent.thread

class MainActivity : ComponentActivity() {

    private var engine: SherpaAsrEngine? = null

    @Volatile
    private var isRecording = false
    private var audioRecord: AudioRecord? = null
    private var workerThread: Thread? = null

    // UI 状态
    private val engineReady = mutableStateOf(false)
    private val recording = mutableStateOf(false)
    private val partialText = mutableStateOf("")
    private val finalResults = mutableStateListOf<String>()
    private val rtfText = mutableStateOf("")

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startRecording()
            } else {
                Toast.makeText(this, "需要麦克风权限才能识别", Toast.LENGTH_LONG).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 模型加载约需数秒，放后台线程，避免 ANR
        thread(name = "asr-init") {
            val e = SherpaAsrEngine(assets)
            runOnUiThread {
                engine = e
                engineReady.value = true
            }
        }

        setContent {
            MaterialTheme {
                AsrScreen()
            }
        }
    }

    @androidx.compose.runtime.Composable
    private fun AsrScreen() {
        val ready by engineReady
        val rec by recording
        val partial by partialText
        val rtf by rtfText
        val listState = rememberLazyListState()

        // 新结果到来时自动滚动到底部
        LaunchedEffect(finalResults.size, partial) {
            val total = finalResults.size + 1
            listState.animateScrollToItem((total - 1).coerceAtLeast(0))
        }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text("中英文混合实时语音识别", fontSize = 20.sp)
            Spacer(Modifier.height(4.dp))
            Text(
                if (rtf.isBlank()) "sherpa-onnx · Zipformer · 端侧推理"
                else "sherpa-onnx · Zipformer · $rtf",
                fontSize = 12.sp,
                color = MaterialTheme.colorScheme.outline,
            )
            Spacer(Modifier.height(12.dp))

            LazyColumn(
                state = listState,
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                items(finalResults) { line ->
                    Card(modifier = Modifier.fillMaxWidth()) {
                        Text(line, modifier = Modifier.padding(12.dp), fontSize = 16.sp)
                    }
                }
                if (partial.isNotBlank()) {
                    item {
                        Card(modifier = Modifier.fillMaxWidth()) {
                            Text(
                                partial,
                                modifier = Modifier.padding(12.dp),
                                fontSize = 16.sp,
                                color = MaterialTheme.colorScheme.primary,
                            )
                        }
                    }
                }
            }

            Spacer(Modifier.height(16.dp))

            if (!ready) {
                CircularProgressIndicator()
                Spacer(Modifier.height(8.dp))
                Text("正在加载模型…", fontSize = 14.sp)
            } else {
                Button(
                    onClick = { if (rec) stopRecording() else checkPermissionAndStart() },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(if (rec) "停止识别" else "开始识别", fontSize = 18.sp)
                }
            }
            Spacer(Modifier.height(8.dp))
        }
    }

    private fun checkPermissionAndStart() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startRecording()
        } else {
            permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    @SuppressLint("MissingPermission")
    private fun startRecording() {
        val e = engine ?: return
        if (isRecording) return

        val minBuf = AudioRecord.getMinBufferSize(
            SherpaAsrEngine.SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        val record = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            SherpaAsrEngine.SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minBuf, SherpaAsrEngine.SAMPLE_RATE / 2), // >= 0.25s 缓冲
        )
        if (record.state != AudioRecord.STATE_INITIALIZED) {
            record.release()
            Toast.makeText(this, "麦克风初始化失败", Toast.LENGTH_LONG).show()
            return
        }

        audioRecord = record
        isRecording = true
        recording.value = true
        // 识别期间保持屏幕常亮
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        record.startRecording()

        workerThread = thread(name = "asr-worker") {
            val buffer = ShortArray(SherpaAsrEngine.SAMPLE_RATE / 10) // 100ms
            // RTF 统计：累计推理耗时 / 累计音频时长
            var totalDecodeNanos = 0L
            var totalAudioSamples = 0L
            var lastRtfUpdate = 0L
            while (isRecording) {
                val n = audioRecord?.read(buffer, 0, buffer.size) ?: break
                if (n <= 0) continue
                val samples = FloatArray(n) { buffer[it] / 32768.0f }

                val t0 = System.nanoTime()
                val (text, isEndpoint) = e.feed(samples)
                totalDecodeNanos += System.nanoTime() - t0
                totalAudioSamples += n

                // 每 500ms 刷新一次 RTF 显示
                val now = System.nanoTime()
                val rtfStr = if (now - lastRtfUpdate > 500_000_000L) {
                    lastRtfUpdate = now
                    val audioSec = totalAudioSamples.toDouble() / SherpaAsrEngine.SAMPLE_RATE
                    val decodeSec = totalDecodeNanos / 1e9
                    "RTF %.3f".format(decodeSec / audioSec)
                } else null

                val processed = TextPostProcessor.process(text)
                // 仅对定稿句子做标点恢复（约几十毫秒），partial 不做以保证实时性
                val display = if (isEndpoint) {
                    TextPostProcessor.polishPunctuation(e.addPunctuation(processed))
                } else {
                    processed
                }
                runOnUiThread {
                    rtfStr?.let { rtfText.value = it }
                    if (isEndpoint) {
                        if (display.isNotBlank()) finalResults.add(display)
                        partialText.value = ""
                    } else {
                        partialText.value = display
                    }
                }
            }
            // 冲刷尾部音频
            val tail = TextPostProcessor.polishPunctuation(
                e.addPunctuation(TextPostProcessor.process(e.finish())),
            )
            runOnUiThread {
                if (tail.isNotBlank()) finalResults.add(tail)
                partialText.value = ""
            }
        }
    }

    private fun stopRecording() {
        if (!isRecording) return
        isRecording = false
        recording.value = false
        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        workerThread?.join(2000)
        workerThread = null
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
    }

    override fun onDestroy() {
        super.onDestroy()
        stopRecording()
        engine?.release()
        engine = null
    }
}
