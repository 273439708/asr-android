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
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import kotlin.concurrent.thread

class MainActivity : ComponentActivity() {

    private var qwenEngine: QwenAsrEngine? = null

    @Volatile
    private var isRecording = false
    private var audioRecord: AudioRecord? = null
    private var workerThread: Thread? = null

    private val engineReady = mutableStateOf(false)
    private val recording = mutableStateOf(false)
    private val partialText = mutableStateOf("")
    private val finalResults = mutableStateListOf<String>()
    private val rtfText = mutableStateOf("")

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startRecording() else Toast.makeText(this, "需要权限", Toast.LENGTH_SHORT).show()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        thread(name = "asr-init") {
            try {
                // 1. 初始化 Qwen 引擎
                // ONNX Runtime 会在第一次使用时自动加载必要的 native 库
                qwenEngine = QwenAsrEngine(this@MainActivity)

                runOnUiThread { engineReady.value = true }
            } catch (e: Throwable) {
                e.printStackTrace()
                val errorMsg = e.localizedMessage ?: e.toString()
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "初始化失败: $errorMsg", Toast.LENGTH_LONG).show()
                }
            }
        }
        setContent { MaterialTheme { AsrScreen() } }
    }

    @Composable
    private fun AsrScreen() {
        val ready by engineReady
        val rec by recording
        val partial by partialText
        val rtf by rtfText
        val listState = rememberLazyListState()

        LaunchedEffect(finalResults.size, partial) {
            val total = finalResults.size + 1
            listState.animateScrollToItem((total - 1).coerceAtLeast(0))
        }

        Column(modifier = Modifier.fillMaxSize().padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Text("Qwen3-ASR 端侧实测", fontSize = 20.sp)
            
            Text(if (rtf.isBlank()) "推理引擎: Qwen3-0.6B" else "引擎: Qwen3-0.6B · $rtf", fontSize = 12.sp, color = MaterialTheme.colorScheme.outline)
            
            LazyColumn(state = listState, modifier = Modifier.weight(1f).fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                items(finalResults) { line -> Card(modifier = Modifier.fillMaxWidth()) { Text(line, modifier = Modifier.padding(12.dp), fontSize = 16.sp) } }
                if (partial.isNotBlank()) {
                    item { Card(modifier = Modifier.fillMaxWidth()) { Text(partial, modifier = Modifier.padding(12.dp), fontSize = 16.sp, color = MaterialTheme.colorScheme.primary) } }
                }
            }

            if (!ready) {
                CircularProgressIndicator()
                Text("正在加载模型 (0.9GB)...", fontSize = 14.sp)
            } else {
                Button(onClick = { if (rec) stopRecording() else checkPermissionAndStart() }, modifier = Modifier.fillMaxWidth()) {
                    Text(if (rec) "停止并识别" else "开始录音", fontSize = 18.sp)
                }
            }
        }
    }

    private fun checkPermissionAndStart() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) startRecording()
        else permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
    }

    @SuppressLint("MissingPermission")
    private fun startRecording() {
        if (isRecording) return
        val sampleRate = 16000
        val minBuf = AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        audioRecord = AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuf, sampleRate * 2))
        
        isRecording = true
        recording.value = true
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        audioRecord?.startRecording()

        workerThread = thread(name = "asr-worker") {
            val buffer = ShortArray(sampleRate / 10)
            val audioAccumulator = mutableListOf<Float>()
            
            while (isRecording) {
                val n = audioRecord?.read(buffer, 0, buffer.size) ?: break
                if (n <= 0) continue
                val samples = FloatArray(n) { buffer[it] / 32768.0f }
                
                audioAccumulator.addAll(samples.toList())
                runOnUiThread { partialText.value = "正在录音... (${audioAccumulator.size / 16000}s)" }
            }
            
            // 停止后的处理
            runOnUiThread { partialText.value = "Qwen3 正在推理..." }
            val t0 = System.nanoTime()
            val result = qwenEngine!!.transcribe(audioAccumulator.toFloatArray())
            val dt = System.nanoTime() - t0
            runOnUiThread {
                if (result.isNotBlank()) {
                    finalResults.add(TextPostProcessor.process(result))
                }
                partialText.value = ""
                rtfText.value = "RTF %.3f".format((dt / 1e9) / (audioAccumulator.size / 16000.0))
            }
        }
    }

    private fun stopRecording() {
        isRecording = false
        recording.value = false
        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
    }
}
