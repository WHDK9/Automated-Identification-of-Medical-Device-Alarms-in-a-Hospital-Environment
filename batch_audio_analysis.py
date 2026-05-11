# D:\ls\NUIG\Design\batch_audio_analysis.py
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import json

class MedicalAlarmAnalyzer:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.alarms_data = {}
        
    def analyze_all_alarms(self, audio_dir, output_dir):
        """Analyze all alarm audio files in the directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all .wav files
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        audio_files.sort()  # Ensure sorted in numerical order
        
        print(f"Found {len(audio_files)} audio files: {audio_files}")
        
        for audio_file in audio_files:
            alarm_id = audio_file.replace('.wav', '')
            file_path = os.path.join(audio_dir, audio_file)
            
            print(f"Analyzing {audio_file}...")
            features = self.analyze_single_alarm(file_path, alarm_id, output_dir)
            self.alarms_data[alarm_id] = features
        
        # Save analysis results
        self.save_analysis_report(output_dir)
        self.generate_comparison_charts(output_dir)
        
        return self.alarms_data
    
    def analyze_single_alarm(self, file_path, alarm_id, output_dir):
        """Analyze a single alarm audio file"""
        # Load audio
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        
        features = {
            'alarm_id': alarm_id,
            'duration': len(y) / sr,
            'sample_rate': sr,
            'file_path': file_path
        }
        
        # Basic time-domain features
        features.update(self.extract_time_domain_features(y))
        
        # Frequency-domain features
        features.update(self.extract_frequency_domain_features(y, sr))
        
        # Medical alarm-specific features
        features.update(self.extract_medical_features(y, sr))
        
        # Generate visualizations
        self.generate_visualizations(y, sr, alarm_id, output_dir, features)
        
        return features
    
    def extract_time_domain_features(self, y):
        """Extract time-domain features"""
        features = {}
        
        # Amplitude statistics
        features['rms'] = np.mean(librosa.feature.rms(y=y))
        features['max_amplitude'] = np.max(np.abs(y))
        features['dynamic_range'] = np.max(y) - np.min(y)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zero_crossing_rate'] = np.mean(zcr)
        
        # Pulse detection
        features.update(self.detect_pulse_patterns(y))
        
        return features
    
    def extract_frequency_domain_features(self, y, sr):
        """Extract frequency-domain features"""
        features = {}
        
        # Spectral features
        stft = np.abs(librosa.stft(y))
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(S=stft, sr=sr))
        features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(S=stft, sr=sr))
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(S=stft, sr=sr))
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
        for i in range(spectral_contrast.shape[0]):
            features[f'spectral_contrast_band_{i}'] = np.mean(spectral_contrast[i])
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        
        # Fundamental frequency detection
        features.update(self.detect_fundamental_frequency(y, sr))
        
        return features
    
    def extract_medical_features(self, y, sr):
        """Extract medical alarm-specific features"""
        features = {}
        
        # Harmonic analysis
        harmonic, percussive = librosa.effects.harmonic(y), librosa.effects.percussive(y)
        features['harmonic_ratio'] = np.sum(harmonic**2) / (np.sum(harmonic**2) + np.sum(percussive**2))
        
        # Pulse repetition pattern
        features.update(self.analyze_pulse_repetition(y, sr))
        
        # Frequency stability
        features.update(self.analyze_frequency_stability(y, sr))
        
        return features
    
    def detect_pulse_patterns(self, y):
        """Detect pulse patterns"""
        features = {}
        
        # Envelope detection using Hilbert transform
        envelope = np.abs(signal.hilbert(y))
        
        # Find peaks (pulses)
        peaks, properties = signal.find_peaks(envelope, height=np.mean(envelope)*0.5, distance=100)
        
        if len(peaks) > 1:
            # Calculate pulse intervals
            peak_intervals = np.diff(peaks) / self.sample_rate
            features['pulse_count'] = len(peaks)
            features['pulse_rate'] = len(peaks) / (len(y) / self.sample_rate)  # pulses per second
            features['pulse_interval_mean'] = np.mean(peak_intervals)
            features['pulse_interval_std'] = np.std(peak_intervals)
        else:
            features['pulse_count'] = 0
            features['pulse_rate'] = 0
            features['pulse_interval_mean'] = 0
            features['pulse_interval_std'] = 0
        
        return features
    
    def detect_fundamental_frequency(self, y, sr):
        """Detect fundamental frequency"""
        features = {}
        
        try:
            # Detect fundamental frequency using autocorrelation method
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=1000, sr=sr)
            f0 = f0[~np.isnan(f0)]  # Remove NaN values
            
            if len(f0) > 0:
                features['fundamental_freq_mean'] = np.mean(f0)
                features['fundamental_freq_std'] = np.std(f0)
                features['fundamental_freq_range'] = np.max(f0) - np.min(f0)
            else:
                features['fundamental_freq_mean'] = 0
                features['fundamental_freq_std'] = 0
                features['fundamental_freq_range'] = 0
        except:
            features['fundamental_freq_mean'] = 0
            features['fundamental_freq_std'] = 0
            features['fundamental_freq_range'] = 0
        
        return features
    
    def analyze_pulse_repetition(self, y, sr):
        """Analyze pulse repetition patterns"""
        features = {}
        
        # Compute short-time energy
        frame_length = 512
        hop_length = 256
        energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Detect energy peaks
        energy_peaks, _ = signal.find_peaks(energy, height=np.mean(energy)*0.7)
        
        if len(energy_peaks) > 1:
            # Calculate repetition period
            repetition_intervals = np.diff(energy_peaks) * hop_length / sr
            features['repetition_rate'] = 1.0 / np.mean(repetition_intervals) if np.mean(repetition_intervals) > 0 else 0
            features['repetition_consistency'] = 1.0 - (np.std(repetition_intervals) / np.mean(repetition_intervals))
        else:
            features['repetition_rate'] = 0
            features['repetition_consistency'] = 0
        
        return features
    
    def analyze_frequency_stability(self, y, sr):
        """Analyze frequency stability"""
        features = {}
        
        # Compute spectral centroid variation over time
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['frequency_stability'] = 1.0 - (np.std(spectral_centroids) / np.mean(spectral_centroids))
        
        return features
    
    def generate_visualizations(self, y, sr, alarm_id, output_dir, features):
        """Generate visualization charts for each alarm"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Medical Alarm {alarm_id} Analysis', fontsize=16)
        
        # 1. Waveform
        librosa.display.waveshow(y, sr=sr, ax=axes[0,0])
        axes[0,0].set_title(f'Waveform (Duration: {features["duration"]:.2f}s)')
        axes[0,0].set_xlabel('Time (s)')
        axes[0,0].set_ylabel('Amplitude')
        
        # 2. Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=axes[0,1])
        axes[0,1].set_title('Spectrogram')
        fig.colorbar(img, ax=axes[0,1], format='%+2.0f dB')
        
        # 3. MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        img = librosa.display.specshow(mfccs, x_axis='time', ax=axes[1,0])
        axes[1,0].set_title('MFCC Features')
        fig.colorbar(img, ax=axes[1,0])
        
        # 4. Feature summary
        axes[1,1].axis('off')
        feature_text = (
            f"Duration: {features['duration']:.2f}s"
            f"Fundamental Freq: {features.get('fundamental_freq_mean', 0):.1f}Hz"
            f"Spectral Centroid: {features['spectral_centroid']:.1f}Hz"
            f"Pulse Rate: {features.get('pulse_rate', 0):.1f}Hz"
            f"RMS Energy: {features['rms']:.4f}"
            f"Zero Crossing: {features['zero_crossing_rate']:.4f}"
            f"Harmonic Ratio: {features.get('harmonic_ratio', 0):.3f}"
        )
        axes[1,1].text(0.1, 0.9, feature_text, transform=axes[1,1].transAxes, 
                      fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'alarm_{alarm_id}_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_analysis_report(self, output_dir):
        report_path = os.path.join(output_dir, 'analysis_report.json')
    
        # Convert NumPy types to native Python types
        def convert_to_serializable(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        # Note: this line must be inside the function with correct indentation
        serializable_data = convert_to_serializable(self.alarms_data)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        print(f"Analysis report saved to: {report_path}")
    
    def generate_comparison_charts(self, output_dir):
        """Generate alarm comparison charts"""
        if not self.alarms_data:
            return
        
        # Create feature comparison chart
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        alarm_ids = list(self.alarms_data.keys())
        
        # 1. Fundamental frequency comparison
        fundamental_freqs = [self.alarms_data[aid].get('fundamental_freq_mean', 0) for aid in alarm_ids]
        axes[0,0].bar(alarm_ids, fundamental_freqs)
        axes[0,0].set_title('Fundamental Frequency Comparison')
        axes[0,0].set_ylabel('Frequency (Hz)')
        
        # 2. Spectral centroid comparison
        spectral_centroids = [self.alarms_data[aid]['spectral_centroid'] for aid in alarm_ids]
        axes[0,1].bar(alarm_ids, spectral_centroids, color='orange')
        axes[0,1].set_title('Spectral Centroid Comparison')
        axes[0,1].set_ylabel('Frequency (Hz)')
        
        # 3. Pulse rate comparison
        pulse_rates = [self.alarms_data[aid].get('pulse_rate', 0) for aid in alarm_ids]
        axes[0,2].bar(alarm_ids, pulse_rates, color='green')
        axes[0,2].set_title('Pulse Rate Comparison')
        axes[0,2].set_ylabel('Pulses per Second')
        
        # 4. Duration comparison
        durations = [self.alarms_data[aid]['duration'] for aid in alarm_ids]
        axes[1,0].bar(alarm_ids, durations, color='red')
        axes[1,0].set_title('Duration Comparison')
        axes[1,0].set_ylabel('Duration (s)')
        
        # 5. RMS energy comparison
        rms_energies = [self.alarms_data[aid]['rms'] for aid in alarm_ids]
        axes[1,1].bar(alarm_ids, rms_energies, color='purple')
        axes[1,1].set_title('RMS Energy Comparison')
        axes[1,1].set_ylabel('RMS Energy')
        
        # 6. Harmonic ratio comparison
        harmonic_ratios = [self.alarms_data[aid].get('harmonic_ratio', 0) for aid in alarm_ids]
        axes[1,2].bar(alarm_ids, harmonic_ratios, color='brown')
        axes[1,2].set_title('Harmonic Ratio Comparison')
        axes[1,2].set_ylabel('Harmonic Ratio')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'alarm_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self, output_dir, df):
        """Generate summary report"""
        report = {
            "analysis_summary": {
                "total_alarms_analyzed": len(df),
                "average_duration": df['duration'].mean(),
                "duration_range": f"{df['duration'].min():.2f}-{df['duration'].max():.2f}s",
                "frequency_range": f"{df['spectral_centroid'].min():.0f}-{df['spectral_centroid'].max():.0f}Hz"
            },
            "alarm_categories": self.categorize_alarms(df),
            "recommendations": self.generate_recommendations(df)
        }
        
        with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
            json.dump(report, f, indent=2)
    
    def categorize_alarms(self, df):
        """Categorize alarms"""
        categories = {}
        
        # Categorize by pulse pattern
        pulsed_alarms = df[df['pulse_rate'] > 2].index.tolist()
        continuous_alarms = df[df['pulse_rate'] <= 2].index.tolist()
        
        categories['pulsed_alarms'] = pulsed_alarms
        categories['continuous_alarms'] = continuous_alarms
        
        # Categorize by frequency
        low_freq = df[df['fundamental_freq_mean'] < 300].index.tolist()
        mid_freq = df[(df['fundamental_freq_mean'] >= 300) & (df['fundamental_freq_mean'] < 800)].index.tolist()
        high_freq = df[df['fundamental_freq_mean'] >= 800].index.tolist()
        
        categories['low_frequency'] = low_freq
        categories['medium_frequency'] = mid_freq
        categories['high_frequency'] = high_freq
        
        return categories
    
    def generate_recommendations(self, df):
        """Generate processing recommendations"""
        recommendations = []
        
        # Recommendations based on pulse rate
        if df['pulse_rate'].max() > 5:
            recommendations.append("High pulse rate alarms may require special temporal pattern recognition")
        
        # Recommendations based on frequency stability
        freq_stable = df[df.get('frequency_stability', 0) > 0.9]
        if len(freq_stable) > 0:
            recommendations.append(f"{len(freq_stable)} alarms show high frequency stability - good for frequency-based classification")
        
        # Recommendations based on duration
        short_alarms = df[df['duration'] < 1.0]
        if len(short_alarms) > 0:
            recommendations.append(f"{len(short_alarms)} short-duration alarms may require overlapping window processing")
        
        return recommendations

# Usage example
if __name__ == "__main__":
    # Configure paths
    audio_directory = "Sounds/Alarms-20251113T232529Z-1-001/Alarms"  # Directory containing 0.wav, 1.wav, ..., 9.wav
    output_directory = "Sounds/analysis_results"
    
    # Create analyzer and run analysis
    analyzer = MedicalAlarmAnalyzer(sample_rate=16000)
    results = analyzer.analyze_all_alarms(audio_directory, output_directory)
    
    print(f"Analysis complete! Results saved to {output_directory}")
    print(f"Analyzed {len(results)} alarm files")
