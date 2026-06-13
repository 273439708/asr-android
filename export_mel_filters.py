"""
导出 Qwen3-ASR(WhisperFeatureExtractor) 的 128x201 slaney mel 滤波器组，
供 Android MelFeatureExtractor 加载(asset: qwen3-asr/mel_filters_128x201.bin)。

布局: float32, little-endian, mel 优先(行=128 mel, 列=201 freq), 行主序。
与 transformers WhisperFeatureExtractor.mel_filters (形状 [201,128]) 转置后一致。

用法:
  python export_mel_filters.py
"""
import numpy as np
from transformers import WhisperFeatureExtractor

MODEL = "M:/models/Qwen3-ASR-0.6B"
OUT = "app/src/main/assets/qwen3-asr/mel_filters_128x201.bin"

fe = WhisperFeatureExtractor.from_pretrained(MODEL)
mf = np.asarray(fe.mel_filters)               # [201, 128] (freq, mel)
mf_mel_major = mf.T.astype(np.float32).copy()  # [128, 201] (mel, freq)
assert mf_mel_major.shape == (128, 201), mf_mel_major.shape
mf_mel_major.tofile(OUT)
print(f"Done. n_fft={fe.n_fft} hop={fe.hop_length} feature_size={fe.feature_size}")
print(f"Wrote {mf_mel_major.shape} float32 -> {OUT} ({mf_mel_major.nbytes} bytes)")
