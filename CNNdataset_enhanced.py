# CNNdataset_enhanced.py
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

class EnhancedSpectrogramGenerator:
    """
    针对各类警报特征生成多通道频谱图
    Channel 0: 标准STFT频谱图 (0-4000Hz)
    Channel 1: 谐波增强频谱图
    Channel 2: 时序变化率图 (突出周期性/瞬变)
    """
    def __init__(self, sr=16000, n_fft=1024, hop_length=512):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length  # 50% overlap
        self.fmax = 4000
    
    def generate_stft(self, audio):
        """标准STFT，对应FFT参数"""
        D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        # 只取0-4000Hz部分
        freq_bins = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        mask = freq_bins <= self.fmax
        D_limited = D[mask, :]
        magnitude = np.abs(D_limited)
        log_mag = librosa.amplitude_to_db(magnitude, ref=np.max)
        return log_mag
    
    def generate_harmonic_enhanced(self, audio):
        """谐波增强图，突出基频和谐波结构"""
        D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        harmonic, _ = librosa.decompose.hpss(D)
        freq_bins = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        mask = freq_bins <= self.fmax
        H_limited = harmonic[mask, :]
        magnitude = np.abs(H_limited)
        log_mag = librosa.amplitude_to_db(magnitude, ref=np.max)
        return log_mag
    
    def generate_temporal_delta(self, stft_db):
        """时序差分图，突出周期性变化和瞬态"""
        delta = librosa.feature.delta(stft_db)
        return delta
    
    def generate_3channel_image(self, audio_path, target_size=(224, 224)):
        """生成3通道融合频谱图"""
        audio, _ = librosa.load(audio_path, sr=self.sr)
        
        ch0 = self.generate_stft(audio)
        ch1 = self.generate_harmonic_enhanced(audio)
        ch2 = self.generate_temporal_delta(ch0)
        
        def normalize(x):
            x = x - x.min()
            if x.max() > 0:
                x = x / x.max()
            return (x * 255).astype(np.uint8)
        
        from PIL import Image as PILImage
        import cv2
        
        h, w = target_size
        
        c0 = cv2.resize(normalize(ch0), (w, h))
        c1 = cv2.resize(normalize(ch1), (w, h))
        c2 = cv2.resize(normalize(ch2), (w, h))
        
        rgb = np.stack([c0, c1, c2], axis=2)
        return PILImage.fromarray(rgb)
