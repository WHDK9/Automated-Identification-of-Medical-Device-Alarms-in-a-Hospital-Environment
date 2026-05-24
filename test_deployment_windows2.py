# test_deployment_windows2.py
import os
import torch
import numpy as np
import sounddevice as sd
import librosa
import cv2
from PIL import Image
import time
from datetime import datetime
from collections import deque
import json
import threading
import winsound
from torchvision import transforms

# 从训练文件导入一致的生成器
from CNNtrain_full import EnhancedSpectrogramGenerator
from CNNmodel_enhanced import AlarmCNN_Enhanced
from CNNrule_checker import AlarmRuleChecker

# ==================== 各类别独立阈值 ====================
CLASS_THRESHOLDS = {
    0: 0.70,
    1: 0.50,
    2: 0.50,   
    3: 0.50,
    4: 0.50,
    5: 0.50,
    6: 0.50,
    7: 0.50,  
    8: 0.50,
    9: 0.50,
}

class WindowsAlarmSystem:
    def __init__(self,
                 model_path='best_cnn_enhanced.pth',
                 sample_rate=16000,
                 analysis_duration=2.0,
                 confidence_threshold=0.50,
                 rule_weight=0.3):

        print("=" * 70)
        print(" Initializing Windows Audio Alarm Recognition System")
        print("=" * 70)

        self.sample_rate = sample_rate
        self.analysis_duration = analysis_duration
        self.confidence_threshold = confidence_threshold
        self.rule_weight = rule_weight

        self.chunk_size = int(sample_rate * analysis_duration)

        # ---- 线程安全的音频缓冲区 ----
        self._buffer_lock = threading.Lock()
        self.audio_buffer = np.zeros(self.chunk_size, dtype=np.float32)

        # ---- 防止推理线程堆积 ----
        self._processing = False

        # ---- 与训练完全一致的频谱图生成器 ----
        self.generator = EnhancedSpectrogramGenerator(
            sr=sample_rate,
            n_fft=1024,
            hop_length=512,
            fmax=4000
        )

        # ---- 模型加载 ----
        print("Loading model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        print(f"Model loaded (Device: {self.device})")

        # ---- 与训练一致的预处理 ----
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # ---- 规则核验器 ----
        self.checker = AlarmRuleChecker(
            sr=sample_rate,
            n_fft=1024,
            hop_length=512
        )

        self.class_names = [f'alarm_{i}' for i in range(10)]
        self.is_running = False
        self.prediction_history = deque(maxlen=3)
        self.alarm_history = []
        self.inference_times = []
        self.audio_levels = []

        self._list_audio_devices()
        print("System initialization completed")

    # ------------------------------------------------------------------
    def _load_model(self, model_path):
        """加载 AlarmCNN_Enhanced 权重"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        model = AlarmCNN_Enhanced(num_classes=10)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        print(f"  Weight has been loaded: {model_path}")
        return model

    # ------------------------------------------------------------------
    def _list_audio_devices(self):
        print("Available audio devices:")
        print("-" * 70)
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                print(f"  [{i}] {dev['name']}")
                print(f"      Channels: {dev['max_input_channels']}, "
                      f"SR: {dev['default_samplerate']}")
        print("-" * 70)
        default = sd.query_devices(kind='input')
        print(f"Default input: {default['name']}")

    # ------------------------------------------------------------------
    def start(self, device_id=None):
        print("Starting system...")
        self.is_running = True
        self.stream = sd.InputStream(
            device=device_id,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size // 4,
            dtype='float32',
            callback=self._audio_callback
        )
        self.stream.start()
        print("System started — Press Ctrl+C to stop")
        print("=" * 70)

    def stop(self):
        print("Stopping system...")
        self.is_running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        self._print_statistics()
        print("System stopped")

    # ------------------------------------------------------------------
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
        if not self.is_running:
            return

        audio_chunk = indata[:, 0].astype(np.float32)

        with self._buffer_lock:
            self.audio_buffer = np.roll(self.audio_buffer, -len(audio_chunk))
            self.audio_buffer[-len(audio_chunk):] = audio_chunk

        # 防止推理线程堆积
        if not self._processing:
            threading.Thread(target=self._process_audio, daemon=True).start()

    # ------------------------------------------------------------------
    def _process_audio(self):
        self._processing = True
        try:
            with self._buffer_lock:
                audio_snap = self.audio_buffer.copy()

            rms = float(np.sqrt(np.mean(audio_snap ** 2)))
            self.audio_levels.append(rms)

            if rms < 0.005:
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[{timestamp}] Silence (RMS:{rms:.4f})", end='\r')
                return

            start_time = time.time()

            # ---- 与训练完全一致的三通道频谱图 ----
            image = self.generator.audio_to_image(audio_snap, target_size=(224, 224))

            # ---- 模型推理 ----
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                model_probs = torch.sigmoid(logits).squeeze().cpu().numpy()

            # ---- 规则融合 ----
            fused = self.checker.fuse_with_model(
                model_probs, audio_snap, rule_weight=self.rule_weight
            )

            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)

            # ---- 应用各类别独立阈值 ----
            preds = np.zeros(10, dtype=int)
            for i in range(10):
                if fused[i] > CLASS_THRESHOLDS[i]:
                    preds[i] = 1

            # ---- alarm_7 冲突消解 ----
            preds = self._resolve_conflicts(preds, fused)

            result = {
                'probabilities': fused,
                'predictions': preds,
                'is_noise': (preds.sum() == 0)
            }

            self.prediction_history.append(result)
            smoothed = self._smooth_predictions()
            self._handle_predictions(smoothed, rms, inference_time)

        except Exception as e:
            print(f"Processing error: {e}")
        finally:
            self._processing = False

    # ------------------------------------------------------------------
    def _resolve_conflicts(self, preds, probs):
        """冲突消解逻辑"""
        # alarm_2 vs alarm_7
        if preds[2] == 1 and preds[7] == 1:
            if probs[7] > probs[2]:
                preds[2] = 0
            else:
                preds[7] = 0

        # alarm_7 被 alarm_3/4 干扰
        if preds[7] == 1 and (preds[3] == 1 or preds[4] == 1):
            if probs[7] < 0.90:
                preds[7] = 0

        # alarm_7 被 alarm_8 覆盖
        if preds[7] == 1 and preds[8] == 1 and probs[8] > probs[7]:
            preds[7] = 0

        return preds

    # ------------------------------------------------------------------
    def _smooth_predictions(self):
        """多帧平滑投票"""
        if len(self.prediction_history) < 2:
            return self.prediction_history[-1]

        recent_preds = [p['predictions'] for p in self.prediction_history]
        recent_probs = [p['probabilities'] for p in self.prediction_history]
        vote_counts  = np.sum(recent_preds, axis=0)

        smoothed_preds = (vote_counts >= len(recent_preds) / 2).astype(int)

        # alarm_7 需全帧确认
        if smoothed_preds[7] == 1 and vote_counts[7] < len(recent_preds):
            smoothed_preds[7] = 0

        return {
            'probabilities': np.mean(recent_probs, axis=0),
            'predictions':   smoothed_preds,
            'is_noise':      (smoothed_preds.sum() == 0)
        }

    # ------------------------------------------------------------------
    def _handle_predictions(self, result, rms, inference_time):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        if result['is_noise']:
            print(f"[{timestamp}] Background  RMS:{rms:.4f}  "
                  f"Inf:{inference_time*1000:.1f}ms", end='\r')
            return

        detected = [
            (i, result['probabilities'][i])
            for i in range(10)
            if result['predictions'][i] == 1
            and result['probabilities'][i] > self.confidence_threshold
        ]

        if not detected:
            return

        print(f"{'='*70}")
        print(f"  ⚠  Alarm Detected  [{timestamp}]")
        print(f"{'='*70}")
        for alarm_id, conf in detected:
            print(f"   {self.class_names[alarm_id]}: {conf:.2%}")
        print(f"  RMS: {rms:.4f}  |  Inference: {inference_time*1000:.1f}ms")
        print(f"{'='*70}")

        threading.Thread(target=lambda: winsound.Beep(800, 200),
                         daemon=True).start()

        self.alarm_history.append({
            'timestamp':      timestamp,
            'alarms':         [(self.class_names[aid], float(c)) for aid, c in detected],
            'rms':            float(rms),
            'inference_time': float(inference_time)
        })

    # ------------------------------------------------------------------
    def _print_statistics(self):
        print("" + "="*70)
        print(" Statistics")
        print("="*70)
        if self.inference_times:
            t = np.array(self.inference_times)
            print(f"Inference  avg:{t.mean()*1000:.1f}ms  "
                  f"min:{t.min()*1000:.1f}ms  max:{t.max()*1000:.1f}ms  "
                  f"FPS:{1/t.mean():.1f}")
        if self.audio_levels:
            lv = np.array(self.audio_levels)
            print(f"Audio      avg:{lv.mean():.4f}  max:{lv.max():.4f}")
        print(f"Alarms detected: {len(self.alarm_history)}")
        for a in self.alarm_history[-5:]:
            print(f"  [{a['timestamp']}] {a['alarms']}")
        print("="*70)

    def save_log(self, filename='alarm_log.json'):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.alarm_history, f, indent=2, ensure_ascii=False)
        print(f"Log saved: {filename}")


# ==================== 主程序 ====================
if __name__ == '__main__':
    system = WindowsAlarmSystem(
        model_path='best_cnn_enhanced.pth',
        sample_rate=16000,
        analysis_duration=2.0,
        confidence_threshold=0.50,
        rule_weight=0.3
    )

    try:
        system.start(device_id=None)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stop signal received...")
    finally:
        system.stop()
        if system.alarm_history:
            system.save_log()
