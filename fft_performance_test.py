import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

class AudioFFTAnalyzer:
    def __init__(self, audio_folder='Sounds/Alarms-20251113T232529Z-1-001/Alarms'):
        self.audio_folder = audio_folder
        self.audio_data = {}

    def load_audio_files(self):
        print(f"Loading audio files from: {self.audio_folder}")
        loaded_count = 0
        for i in range(0, 10):
            filename = f"{i}.wav"
            filepath = os.path.join(self.audio_folder, filename)
            if os.path.exists(filepath):
                try:
                    sr, audio = wavfile.read(filepath)
                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)
                    if audio.dtype == np.int16:
                        audio = audio.astype(np.float32) / 32768.0
                    elif audio.dtype == np.int32:
                        audio = audio.astype(np.float32) / 2147483648.0
                    self.audio_data[i] = {
                        'filename': filename,
                        'sample_rate': sr,
                        'audio': audio,
                        'duration': len(audio) / sr
                    }
                    loaded_count += 1
                    print(f"  Loaded {filename}: {sr}Hz, {len(audio)} samples, {self.audio_data[i]['duration']:.2f}s")
                except Exception as e:
                    print(f"  Error loading {filename}: {e}")
            else:
                print(f"  File not found: {filename}")
        print(f"Total loaded: {loaded_count} files")
        return self.audio_data

    def analyze_fft_in_windows(self, audio_id, window_sec=0.1, overlap=0.5, max_freq=4000, isdetrend=True):
        if audio_id not in self.audio_data:
            print(f"Audio {audio_id} not loaded")
            return None

        data = self.audio_data[audio_id]
        audio = data['audio']
        sr = data['sample_rate']

        window_len = int(sr * window_sec)
        hop_len = int(window_len * (1 - overlap))
        total_len = len(audio)
        window_func = np.hanning(window_len)
        results = []

        for start_idx in range(0, total_len - window_len + 1, hop_len):
            segment = audio[start_idx:start_idx + window_len].copy()
            if isdetrend:
                segment -= np.mean(segment)
            segment *= window_func

            fft_result = np.fft.rfft(segment)
            freqs = np.fft.rfftfreq(len(segment), 1/sr)

            mag = np.abs(fft_result)
            mag = np.maximum(mag, 1e-12)
            magDB = 20 * np.log10(mag)
            magDB -= np.max(magDB)
            magDB = np.clip(magDB, a_min=-120, a_max=None)

            max_idx = np.searchsorted(freqs, max_freq, side='right')
            freqs_plot = freqs[:max_idx]
            spectrum_plot = magDB[:max_idx]

            results.append({
                'window_index': start_idx // hop_len,
                'start_time_ms': (start_idx / sr) * 1000,
                'end_time_ms': ((start_idx + window_len) / sr) * 1000,
                'freqs': freqs_plot,
                'spectrum': spectrum_plot,
                'spectrum_label': 'Amplitude (dB, normalized)',
                'filename': data['filename']
            })

        return results

    def save_fft_windows_images(self, fft_windows, base_save_dir='Sounds/FFT'):
        if not fft_windows:
            print("No FFT windows data to save.")
            return

        # Get audio filename from the first window
        fname = fft_windows[0].get('filename', 'Audio')
        name_wo_ext = os.path.splitext(fname)[0]

        save_folder = os.path.join(base_save_dir, name_wo_ext)
        os.makedirs(save_folder, exist_ok=True)

        for wdata in fft_windows:
            freqs = wdata['freqs']
            spectrum = wdata['spectrum']
            label = wdata['spectrum_label']
            start_t = wdata['start_time_ms']
            end_t = wdata['end_time_ms']
            window_idx = wdata['window_index']

            plt.figure(figsize=(10, 6))
            plt.plot(freqs, spectrum, linewidth=1)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel(label)
            plt.title(f'{fname} | Window {window_idx}: {start_t:.1f} ms - {end_t:.1f} ms')
            plt.grid(True, alpha=0.3)
            plt.xlim(0, freqs[-1] if len(freqs) > 0 else 4000)
            plt.ylim(-130, 5)

            save_path = os.path.join(save_folder, f'window_{window_idx}.png')
            plt.savefig(save_path)
            plt.close()  # Close figure to prevent memory leaks

        print(f"Saved {len(fft_windows)} FFT window images to folder: {save_folder}")


if __name__ == "__main__":
    print("Batch overlapping short-time FFT analysis and image saving")
    print("="*60)

    analyzer = AudioFFTAnalyzer('Sounds/Alarms-20251113T232529Z-1-001/Alarms')
    analyzer.load_audio_files()

    window_sec = 1024 / 16000
    overlap = 0.5

    for audio_id in analyzer.audio_data.keys():
        print(f"Processing audio ID {audio_id} ...")
        fft_windows = analyzer.analyze_fft_in_windows(audio_id=audio_id, window_sec=window_sec, overlap=overlap, max_freq=4000)
        analyzer.save_fft_windows_images(fft_windows, base_save_dir='Sounds/FFT')

    print("All audio files processed successfully!")
