# windows_audio_alarm_test.py
import torch
import numpy as np
import sounddevice as sd
import librosa
from PIL import Image
import time
from datetime import datetime
from collections import deque
import json
import threading
import winsound  # Windows beeper

class WindowsAlarmSystem:
    """
    Windows Audio Alarm Recognition Test System
    - Real-time audio capture
    - Spectrogram generation
    - Model inference
    - Sound alerts (Windows beeper)
    - Strict recognition strategy for alarm_7
    """
    def __init__(self, 
                 model_path='best_cnn_model_quantized_scripted.pt',
                 sample_rate=16000,
                 analysis_duration=2.0,
                 confidence_threshold=0.70):
        
        print("="*70)
        print(" Initializing Windows Audio Alarm Recognition System")
        print("="*70)
        
        self.sample_rate = sample_rate
        self.analysis_duration = analysis_duration
        self.confidence_threshold = confidence_threshold
        
        # Audio parameters
        self.chunk_size = int(sample_rate * analysis_duration)
        self.audio_buffer = np.zeros(self.chunk_size)
        
        # Load model
        print("Loading model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        print(f"Model loaded successfully (Device: {self.device})")
        
        # Data preprocessing
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Class names
        self.class_names = [
            'alarm_0', 'alarm_1', 'alarm_2', 'alarm_3', 'alarm_4',
            'alarm_5', 'alarm_6', 'alarm_7', 'alarm_8', 'alarm_9'
        ]
        
        # Spectrum-aware classifier thresholds
        self.thresholds = {i: 0.5 for i in range(10)}
        self.thresholds[2] = 0.68  # alarm_2 optimized threshold
        self.thresholds[7] = 0.85  # alarm_7 strict threshold 
        
        # Control flags
        self.is_running = False
        self.prediction_history = deque(maxlen=3)
        self.alarm_history = []
        
        # Performance statistics
        self.inference_times = []
        self.audio_levels = []
        
        # Detect audio devices
        self._list_audio_devices()
        
        print("System initialization completed")
    
    def _list_audio_devices(self):
        """
        List available audio devices
        """
        print("Available audio devices:")
        print("-" * 70)
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  [{i}] {device['name']}")
                print(f"      Input channels: {device['max_input_channels']}, "
                      f"Sample rate: {device['default_samplerate']}")
        print("-" * 70)
        
        # Get default input device
        default_device = sd.query_devices(kind='input')
        print(f"Default input device: {default_device['name']}")
        print()
    
    def _load_model(self, model_path):
        """
        Load optimized model
        """
        try:
            # Try to load TorchScript model
            model = torch.jit.load(model_path, map_location=self.device)
            print(f"  TorchScript model loaded: {model_path}")
        except:
            # Fallback to regular model
            from CNNmodel import AlarmCNN
            model = AlarmCNN(num_classes=10)
            model.load_state_dict(torch.load('best_cnn_model_quantized.pth', 
                                            map_location=self.device))
            print(f"  Regular model loaded: best_cnn_model_quantized.pth")
        
        model.eval()
        return model
    
    def start(self, device_id=None):
        """
        Start system
        
        Args:
            device_id: Audio device ID, None means use default device
        """
        print("Starting system...")
        self.is_running = True
        
        # Start audio stream
        print(f"Starting audio capture (Sample rate: {self.sample_rate} Hz)...")
        self.stream = sd.InputStream(
            device=device_id,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size // 4,  # Smaller block size for real-time
            callback=self._audio_callback
        )
        self.stream.start()
        
        print("System started")
        print("Press Ctrl+C to stop system")
        print("="*70)
    
    def stop(self):
        """
        Stop system
        """
        print("Stopping system...")
        self.is_running = False
        
        # Stop audio stream
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        # Print statistics
        self._print_statistics()
        
        print("System stopped")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """
        Audio stream callback
        """
        if status:
            print(f"Audio status: {status}")
        
        if not self.is_running:
            return
        
        # Update audio buffer
        audio_data = indata.flatten()
        self.audio_buffer = np.roll(self.audio_buffer, -len(audio_data))
        self.audio_buffer[-len(audio_data):] = audio_data
        
        # Process audio asynchronously
        if len(self.audio_buffer) >= self.chunk_size:
            threading.Thread(target=self._process_audio, daemon=True).start()
    
    def _process_audio(self):
        """
        Process audio data
        """
        try:
            # Check audio energy
            rms = np.sqrt(np.mean(self.audio_buffer**2))
            self.audio_levels.append(rms)
            
            if rms < 0.01:  # Audio too quiet, skip
                return
            
            # Generate spectrogram
            start_time = time.time()
            spectrogram = self._generate_spectrogram(self.audio_buffer)
            
            # Inference
            predictions = self._predict(spectrogram)
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Smooth predictions
            self.prediction_history.append(predictions)
            smoothed = self._smooth_predictions()
            
            # Handle results
            self._handle_predictions(smoothed, rms, inference_time)
            
        except Exception as e:
            print(f"Processing error: {e}")
    
    def _generate_spectrogram(self, audio_data):
        """
        Generate spectrogram
        """
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=self.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        
        # Convert to decibels
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to 0-255
        mel_spec_norm = ((mel_spec_db - mel_spec_db.min()) / 
                        (mel_spec_db.max() - mel_spec_db.min() + 1e-8) * 255)
        mel_spec_norm = mel_spec_norm.astype(np.uint8)
        
        # Transpose and convert to RGB (fix dimensions)
        mel_spec_norm = mel_spec_norm.T  # Transpose to (time, freq)
        mel_spec_rgb = np.stack([mel_spec_norm] * 3, axis=-1)  # (time, freq, 3)
        
        # Ensure contiguous array
        mel_spec_rgb = np.ascontiguousarray(mel_spec_rgb)
        
        # Create PIL Image
        image = Image.fromarray(mel_spec_rgb, mode='RGB')
        image = image.resize((224, 224), Image.LANCZOS)
        
        return image
    
    def _predict(self, spectrogram):
        """
        Model inference (enhanced alarm_7 recognition strictness)
        """
        # Preprocessing
        image_tensor = self.transform(spectrogram).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = torch.sigmoid(output)[0].cpu().numpy()
        
        # Apply thresholds
        preds = np.zeros(10, dtype=int)
        for i in range(10):
            if probs[i] > self.thresholds[i]:
                preds[i] = 1
        
        # alarm_7 strict handling
        if preds[7] == 1:
            # Increase alarm_7 confidence threshold
            if probs[7] < 0.85:  # Raised from 0.5 to 0.85
                preds[7] = 0
            
            # If alarm_3 or alarm_4 detected (they have 2000Hz component), exclude alarm_7
            if preds[3] == 1 or preds[4] == 1:
                if probs[7] < 0.90:  # Further raised to 0.90
                    preds[7] = 0
            
            # If alarm_8 detected (energy at 1000-1500Hz and 3000-3500Hz), exclude alarm_7
            if preds[8] == 1 and probs[8] > probs[7]:
                preds[7] = 0
        
        # Original conflict resolution (alarm_2 vs alarm_7)
        if preds[2] == 1 and preds[7] == 1:
            if probs[7] > probs[2]:
                preds[2] = 0
            else:
                preds[7] = 0
        
        return {
            'probabilities': probs,
            'predictions': preds,
            'is_noise': (preds.sum() == 0)
        }
    
    def _smooth_predictions(self):
        """
        Smooth predictions (alarm_7 requires more frame confirmation)
        """
        if len(self.prediction_history) < 2:
            return self.prediction_history[-1]
        
        recent_preds = [p['predictions'] for p in self.prediction_history]
        recent_probs = [p['probabilities'] for p in self.prediction_history]
        
        # Voting
        vote_counts = np.sum(recent_preds, axis=0)
        smoothed_preds = (vote_counts >= len(recent_preds) / 2).astype(int)
        
        # alarm_7 requires all 3 frames to be detected for confirmation
        if smoothed_preds[7] == 1:
            if vote_counts[7] < len(recent_preds):  # Must be detected in all frames
                smoothed_preds[7] = 0
        
        smoothed_probs = np.mean(recent_probs, axis=0)
        
        return {
            'probabilities': smoothed_probs,
            'predictions': smoothed_preds,
            'is_noise': (smoothed_preds.sum() == 0)
        }
    
    def _handle_predictions(self, predictions, rms, inference_time):
        """
        Handle prediction results
        """
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        if predictions['is_noise']:
            # Background noise
            print(f"[{timestamp}] Background noise (RMS: {rms:.4f}, "
                  f"Inference: {inference_time*1000:.1f}ms)", end='\r')
        
        else:
            # Alarm detected
            detected_alarms = []
            for i in range(10):
                if predictions['predictions'][i] == 1:
                    confidence = predictions['probabilities'][i]
                    if confidence > self.confidence_threshold:
                        detected_alarms.append((i, confidence))
            
            if detected_alarms:
                # High confidence alarm
                print(f"{'='*70}")
                print(f"  Alarm Detected [{timestamp}]")
                print(f"{'='*70}")
                
                for alarm_id, confidence in detected_alarms:
                    print(f"   {self.class_names[alarm_id]}: {confidence:.2%}")
                
                print(f"  Audio level: {rms:.4f}")
                print(f"  Inference time: {inference_time*1000:.1f}ms")
                print(f"{'='*70}")
                
                # Windows beeper alert
                self._play_alert_sound()
                
                # Record alarm
                self.alarm_history.append({
                    'timestamp': timestamp,
                    'alarms': [(self.class_names[aid], conf) 
                              for aid, conf in detected_alarms],
                    'rms': float(rms),
                    'inference_time': float(inference_time)
                })
    
    def _play_alert_sound(self):
        """
        Play Windows alert sound
        """
        try:
            # Play system beep (800Hz frequency, 200ms duration)
            threading.Thread(target=lambda: winsound.Beep(800, 200), 
                           daemon=True).start()
        except:
            pass
    
    def _print_statistics(self):
        """
        Print statistics
        """
        print("" + "="*70)
        print(" System Statistics")
        print("="*70)
        
        if self.inference_times:
            avg_time = np.mean(self.inference_times)
            print(f"Inference performance:")
            print(f"  Average inference time: {avg_time*1000:.2f} ms")
            print(f"  Minimum inference time: {np.min(self.inference_times)*1000:.2f} ms")
            print(f"  Maximum inference time: {np.max(self.inference_times)*1000:.2f} ms")
            print(f"  Average FPS: {1.0/avg_time:.2f}")
        
        if self.audio_levels:
            avg_level = np.mean(self.audio_levels)
            print(f"Audio statistics:")
            print(f"  Average audio level: {avg_level:.4f}")
            print(f"  Maximum audio level: {np.max(self.audio_levels):.4f}")
        
        print(f"Alarm statistics:")
        print(f"  Total alarms: {len(self.alarm_history)}")
        
        if self.alarm_history:
            print(f"Recent alarms:")
            for alarm in self.alarm_history[-5:]:
                print(f"  [{alarm['timestamp']}] {alarm['alarms']}")
        
        print("="*70)
    
    def save_log(self, filename='windows_alarm_log.json'):
        """
        Save log
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.alarm_history, f, indent=2, ensure_ascii=False)
        print(f"Log saved: {filename}")

# Main program
if __name__ == '__main__':
    # Create system
    system = WindowsAlarmSystem(
        model_path='best_cnn_model_quantized_scripted.pt',
        sample_rate=16000,
        analysis_duration=2.0,
        confidence_threshold=0.5
    )
    
    try:
        # Start system (use default audio device)
        system.start(device_id=None)
        
        # Keep running
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("Stop signal received...")
    
    finally:
        # Stop system
        system.stop()
        
        # Save log
        if system.alarm_history:
            system.save_log()
