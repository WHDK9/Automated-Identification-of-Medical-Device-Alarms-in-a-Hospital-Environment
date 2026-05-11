"""
Medical Alarm Synthetic Data Generator
"""

import itertools
import numpy as np
import librosa
import librosa.display
import random
import matplotlib.pyplot as plt
import soundfile as sf
import pandas as pd
import os
import shutil
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
SAMPLING_RATE = 16000
DURATION = 5
NUM_ALARMS = 10

ALARM_FILES = [fr"Sounds/Alarms-20251113T232529Z-1-001/Alarms/{i}.wav" for i in range(NUM_ALARMS)]
NOISE_FILE = r"Sounds/Alarms-20251113T232529Z-1-001/Alarms/Background_noise.wav"

# ==================== Output Directory Setup ====================
BASE_OUTPUT_FOLDER = "Generator"
OUTPUT_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "output_spectrograms")
SHUFFLED_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "shuffled_spectrograms")
AUDIO_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "synthetic_audio")  # New: save audio files
EXCEL_FILE = os.path.join(BASE_OUTPUT_FOLDER, "synthetic_dataset.xlsx")

# Spectrogram parameters
N_FFT = 2048           # Larger window
HOP_LENGTH = 512       # Unchanged
FMAX = 8000            # Extended frequency range

# Data generation configuration
NUM_SINGLE_ALARM_SAMPLES = 50
NUM_DUAL_ALARM_SAMPLES = 20
NUM_NO_ALARM_SAMPLES = 30

# ==================== Preprocessing ====================
print("\n" + "="*60)
print(" Medical Alarm Recognition System - Synthetic Data Generator")
print("="*60)
response = input(f"Clear and recreate output directory '{BASE_OUTPUT_FOLDER}'? (y/n): ")
if response.lower() == 'y':
    if os.path.exists(BASE_OUTPUT_FOLDER):
        shutil.rmtree(BASE_OUTPUT_FOLDER)
        print(f"Deleted old directory: {BASE_OUTPUT_FOLDER}")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(SHUFFLED_FOLDER, exist_ok=True)
    os.makedirs(AUDIO_FOLDER, exist_ok=True)
    print("Created new directories")
else:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(SHUFFLED_FOLDER, exist_ok=True)
    os.makedirs(AUDIO_FOLDER, exist_ok=True)
    print("Keeping existing files")

# ==================== Alarm Feature Definitions ====================
ALARM_FEATURES = {
    0: {'name': 'Alarm_0', 'f0': 491.8, 'description': 'Spectral depression 2500-3000Hz at 0.6s', 'duration_range': (0.5, 1.0)},
    1: {'name': 'Alarm_1', 'f0': 962.4, 'description': 'Dual-tone: 1000Hz + 2900Hz', 'duration_range': (0.32, 2.33)},
    2: {'name': 'Alarm_2', 'f0': 3000.43, 'description': 'Pure tone 3000Hz, repeated', 'duration_range': (0.8, 1.39)},
    3: {'name': 'Alarm_3', 'f0': 699.9, 'description': 'Harmonic series throughout', 'duration_range': (2.0, 4.0)},
    4: {'name': 'Alarm_4', 'f0': 423.7, 'description': 'Non-periodic harmonics with decay', 'duration_range': (2.0, 4.0)},
    5: {'name': 'Alarm_5', 'f0': 669.6, 'description': 'Triple pulse pattern', 'duration_range': (0.48, 1.36)},
    6: {'name': 'Alarm_6', 'f0': 614.2, 'description': 'Alternating high-low pattern', 'duration_range': (0.3, 1.5)},
    7: {'name': 'Alarm_7', 'f0': 502.6, 'description': 'Stable harmonics with 2000Hz formant', 'duration_range': (1.1, 2.83)},
    8: {'name': 'Alarm_8', 'f0': 584.7, 'description': 'Dual-band: 1000-1500Hz + 3000-3500Hz', 'duration_range': (2.0, 4.0)},
    9: {'name': 'Alarm_9', 'f0': 159.7, 'description': 'Low F0 with prominent 1500Hz harmonic', 'duration_range': (2.0, 4.0)}
}

# ==================== Preload Alarm Audio ====================
def preload_alarms():
    alarm_data = {}
    print("Preloading alarm audio files...")
    for i, file in enumerate(ALARM_FILES):
        if not os.path.exists(file):
            print(f"  Warning: {file} not found, skipping")
            alarm_data[i] = None
            continue
        try:
            alarm_audio, sr = librosa.load(file, sr=SAMPLING_RATE)
            alarm_data[i] = alarm_audio
            print(f"    Alarm {i}: {len(alarm_audio)/SAMPLING_RATE:.2f}s, F0={ALARM_FEATURES[i]['f0']:.1f}Hz")
        except Exception as e:
            print(f"    Failed to load {file}: {e}")
            alarm_data[i] = None
    return alarm_data

alarm_cache = preload_alarms()

# ==================== Data Augmentation Functions ====================
def apply_augmentation(audio, augmentation_type='none'):
    if augmentation_type == 'none':
        return audio
    elif augmentation_type == 'noise':
        noise_level = random.uniform(0.001, 0.005)
        noise = np.random.normal(0, noise_level, len(audio))
        return audio + noise
    elif augmentation_type == 'gain':
        gain_db = random.uniform(-3, 3)
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear
    elif augmentation_type == 'time_stretch':
        rate = random.uniform(0.95, 1.05)
        return librosa.effects.time_stretch(audio, rate=rate)
    elif augmentation_type == 'pitch_shift':
        n_steps = random.uniform(-0.5, 0.5)
        return librosa.effects.pitch_shift(audio, sr=SAMPLING_RATE, n_steps=n_steps)
    return audio

# ==================== Synthetic Audio Generation ====================
def generate_synthetic_audio(alarm_indices=[], augmentation='none'):
    """Generate synthetic audio samples"""
    synthetic_audio = np.zeros(SAMPLING_RATE * DURATION, dtype=np.float32)
    alarm_presence = [0] * (NUM_ALARMS + 1)

    if not alarm_indices:
        # No-alarm sample
        alarm_presence[-1] = 1
    else:
        # Alarm sample
        for alarm_idx in alarm_indices:
            if alarm_cache[alarm_idx] is None:
                continue
            
            alarm_presence[alarm_idx] = 1
            alarm_audio = alarm_cache[alarm_idx].copy()
            alarm_audio = apply_augmentation(alarm_audio, augmentation)
            
            # Ensure enough space to insert the alarm
            alarm_length = len(alarm_audio)
            max_start = max(0, SAMPLING_RATE * DURATION - alarm_length)
            
            if max_start > 0:
                start_position = random.randint(0, max_start)
            else:
                # If alarm is too long, start from beginning and truncate
                start_position = 0
                alarm_length = SAMPLING_RATE * DURATION
            
            # Insert alarm audio
            end_position = min(start_position + alarm_length, SAMPLING_RATE * DURATION)
            actual_length = end_position - start_position
            synthetic_audio[start_position:end_position] += alarm_audio[:actual_length]
    
    # Normalize signal first
    max_signal = np.max(np.abs(synthetic_audio))
    if max_signal > 0:
        synthetic_audio = synthetic_audio / max_signal * 0.7  # Leave headroom for noise
    
    # Add background noise
    if os.path.exists(NOISE_FILE):
        try:
            noise_audio, _ = librosa.load(NOISE_FILE, sr=SAMPLING_RATE)
            
            # Ensure noise is long enough
            if len(noise_audio) > SAMPLING_RATE * DURATION:
                noise_start = random.randint(0, len(noise_audio) - SAMPLING_RATE * DURATION)
                noise_segment = noise_audio[noise_start:noise_start + SAMPLING_RATE * DURATION]
            else:
                # Loop noise if too short
                repeats = (SAMPLING_RATE * DURATION // len(noise_audio)) + 1
                noise_segment = np.tile(noise_audio, repeats)[:SAMPLING_RATE * DURATION]
            
            # Set reasonable SNR
            if alarm_indices:
                snr_db = random.uniform(15, 25)  # Higher SNR when alarm is present
            else:
                snr_db = random.uniform(5, 15)   # Lower SNR for no-alarm samples
            
            signal_power = np.mean(synthetic_audio ** 2)
            noise_power = np.mean(noise_segment ** 2)
            
            if noise_power > 0:
                noise_gain = np.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10))))
                synthetic_audio += noise_segment * noise_gain
        except Exception as e:
            print(f"  Failed to add noise: {e}")
    
    # Final normalization
    max_val = np.max(np.abs(synthetic_audio))
    if max_val > 0:
        synthetic_audio = synthetic_audio / max_val * 0.95
    
    return synthetic_audio, alarm_presence

# ==================== Spectrogram Generation ====================
def generate_spectrogram(audio, file_name, save_audio=False, audio_name=None):
    """Generate and save mel spectrogram"""
    # Validate audio length
    if len(audio) < N_FFT:
        print(f"  Warning: {file_name} audio too short, skipping")
        return False
    
    # Validate audio energy
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-6:
        print(f"  Warning: {file_name} audio energy too low, skipping")
        return False
    
    # Optionally save audio file for debugging
    if save_audio and audio_name:
        audio_path = os.path.join(AUDIO_FOLDER, audio_name)
        sf.write(audio_path, audio, SAMPLING_RATE)
    
    try:
        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLING_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=128,
            fmin=0,
            fmax=min(FMAX, SAMPLING_RATE // 2)
        )
        
        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Validate spectrogram
        if np.all(np.isnan(mel_spec_db)) or np.all(np.isinf(mel_spec_db)):
            print(f"  Warning: {file_name} spectrogram invalid, skipping")
            return False
        
        # Plot spectrogram
        plt.figure(figsize=(10, 4), dpi=100)
        librosa.display.specshow(
            mel_spec_db,
            sr=SAMPLING_RATE,
            hop_length=HOP_LENGTH,
            x_axis='time',
            y_axis='mel',
            fmin=0,
            fmax=min(FMAX, SAMPLING_RATE // 2),
            cmap='viridis'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(OUTPUT_FOLDER, file_name)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"  Failed to generate spectrogram {file_name}: {e}")
        plt.close()
        return False

# ==================== Dataset Generation ====================
def generate_dataset():
    print(" " + "="*60)
    print(" Starting medical alarm synthetic dataset generation")
    print("="*60 + " ")

    data = []
    spectrogram_files = []
    sample_index = 0
    augmentation_types = ['none', 'noise', 'gain']

    # 1. Single-alarm samples
    print(f" Step 1: Generating single-alarm samples ({NUM_SINGLE_ALARM_SAMPLES} per class)")
    for alarm_idx in tqdm(range(NUM_ALARMS), desc="Single alarm"):
        if alarm_cache[alarm_idx] is None:
            print(f"    Skipping Alarm {alarm_idx} (file not found)")
            continue
        
        for sample_num in range(NUM_SINGLE_ALARM_SAMPLES):
            aug_type = augmentation_types[sample_num % len(augmentation_types)]
            synthetic_audio, alarm_presence = generate_synthetic_audio([alarm_idx], augmentation=aug_type)
            
            spectrogram_file = f"spectrogram_{sample_index:05d}.png"
            audio_file = f"audio_{sample_index:05d}.wav"
            
            # Save first few audio samples for debugging
            save_audio = (sample_index < 10)
            
            if generate_spectrogram(synthetic_audio, spectrogram_file, save_audio, audio_file):
                spectrogram_files.append(spectrogram_file)
                data.append(alarm_presence)
                sample_index += 1
    
    print(f"   Generated {len(spectrogram_files)} single-alarm samples\n")

    # 2. Dual-alarm combination samples
    print(f" Step 2: Generating dual-alarm combination samples ({NUM_DUAL_ALARM_SAMPLES} per combination)")
    dual_combinations = list(itertools.combinations(range(NUM_ALARMS), 2))
    
    for combo in tqdm(dual_combinations, desc="Dual alarm"):
        if alarm_cache[combo[0]] is None or alarm_cache[combo[1]] is None:
            continue
        
        for sample_num in range(NUM_DUAL_ALARM_SAMPLES):
            aug_type = augmentation_types[sample_num % len(augmentation_types)]
            synthetic_audio, alarm_presence = generate_synthetic_audio(list(combo), augmentation=aug_type)
            
            spectrogram_file = f"spectrogram_{sample_index:05d}.png"
            
            if generate_spectrogram(synthetic_audio, spectrogram_file):
                spectrogram_files.append(spectrogram_file)
                data.append(alarm_presence)
                sample_index += 1
    
    print(f"   Generated dual-alarm samples")

    # 3. No-alarm samples
    print(f" Step 3: Generating no-alarm samples ({NUM_NO_ALARM_SAMPLES} total)")
    for _ in tqdm(range(NUM_NO_ALARM_SAMPLES), desc="No alarm"):
        synthetic_audio, alarm_presence = generate_synthetic_audio([], augmentation='noise')
        
        spectrogram_file = f"spectrogram_{sample_index:05d}.png"
        
        if generate_spectrogram(synthetic_audio, spectrogram_file):
            spectrogram_files.append(spectrogram_file)
            data.append(alarm_presence)
            sample_index += 1
    
    print(f"   Generated {NUM_NO_ALARM_SAMPLES} no-alarm samples")

    # 4. Create DataFrame and shuffle
    print(" Step 4: Creating dataset annotation file")
    columns = [f"alarm_{i}" for i in range(NUM_ALARMS)] + ["no_alarm"]
    df = pd.DataFrame(data, columns=columns)
    df["Original_Spectrogram_File"] = spectrogram_files
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Rename files
    print("    Renaming files...")
    new_filenames = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Renaming"):
        old_name = os.path.join(OUTPUT_FOLDER, row["Original_Spectrogram_File"])
        new_name = os.path.join(SHUFFLED_FOLDER, f"spectrogram_{i:05d}.png")

        if os.path.exists(old_name):
            os.replace(old_name, new_name)
            new_filenames.append(f"spectrogram_{i:05d}.png")
        else:
            new_filenames.append("")

    # Update DataFrame
    df.drop(columns=["Original_Spectrogram_File"], inplace=True)
    df.insert(0, "Spectrogram_File", new_filenames)
    df['Total_Alarms'] = df[[f"alarm_{i}" for i in range(NUM_ALARMS)]].sum(axis=1)

    # Save Excel
    df.to_excel(EXCEL_FILE, index=False)

    print(f"Dataset generation complete!")
    print(f"    Spectrograms saved to: {SHUFFLED_FOLDER}/")
    print(f"    Audio samples saved to: {AUDIO_FOLDER}/ (first 10)")
    print(f"    Annotation file: {EXCEL_FILE}")
    print(f"    Total samples: {len(df)}")
    print(f"    Single-alarm samples: {len(df[df['Total_Alarms'] == 1])}")
    print(f"    Dual-alarm samples: {len(df[df['Total_Alarms'] == 2])}")
    print(f"    No-alarm samples: {len(df[df['Total_Alarms'] == 0])}")

    print("Alarm sample distribution by class:")
    for i in range(NUM_ALARMS):
        count = df[f"alarm_{i}"].sum()
        print(f"   Alarm {i} ({ALARM_FEATURES[i]['name']}): {count} samples")

    return df

# ==================== Dataset Validation ====================
def validate_dataset(df):
    print(" " + "="*60)
    print(" Dataset Validation")
    print("="*60 + "")

    missing_files = []
    for idx, row in df.iterrows():
        file_path = os.path.join(SHUFFLED_FOLDER, row['Spectrogram_File'])
        if not os.path.exists(file_path):
            missing_files.append(row['Spectrogram_File'])

    if missing_files:
        print(f"Found {len(missing_files)} missing files")
        print(f"   First 5: {missing_files[:5]}")
    else:
        print("All spectrogram files present")

    label_columns = [f"alarm_{i}" for i in range(NUM_ALARMS)] + ["no_alarm"]
    invalid_labels = []
    for idx, row in df.iterrows():
        label_sum = row[label_columns].sum()
        if label_sum == 0:
            invalid_labels.append(idx)

    if invalid_labels:
        print(f"Found {len(invalid_labels)} invalid labels")
    else:
        print("All labels valid")

    print(f"Dataset statistics:")
    print(f"   Total samples: {len(df)}")
    print(f"   Feature dimensions: {len(df.columns) - 1}")
    print(f"   Number of classes: {NUM_ALARMS + 1}")

    return len(missing_files) == 0 and len(invalid_labels) == 0

# ==================== Main Program ====================
def main():
    print("\nConfiguration:")
    print(f"   Sample rate: {SAMPLING_RATE} Hz")
    print(f"   Sample duration: {DURATION} seconds")
    print(f"   Alarm classes: {NUM_ALARMS}")
    print(f"   FFT window: {N_FFT} ({N_FFT/SAMPLING_RATE*1000:.1f}ms)")
    print(f"   Hop length: {HOP_LENGTH} ({HOP_LENGTH/SAMPLING_RATE*1000:.1f}ms)")
    print(f"   Overlap: {(1 - HOP_LENGTH/N_FFT)*100:.0f}%")
    print(f"   Max frequency: {FMAX} Hz")

    missing_alarms = [i for i in range(NUM_ALARMS) if alarm_cache[i] is None]
    if missing_alarms:
        print(f"Warning: The following alarm files are missing: {missing_alarms}")
        response = input("\nContinue generating dataset? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled")
            return

    df = generate_dataset()
    is_valid = validate_dataset(df)

    if is_valid:
        print(" Dataset generated and validated successfully!")
        print("\n Tip: First 10 audio samples saved to Generator/synthetic_audio/")
        print("  Play these files to check audio quality")
    else:
        print("Dataset generation complete, but some issues were found - please review")

    print("" + "="*60)
    print(" All tasks complete")
    print("="*60 + "")

if __name__ == '__main__':
    main()
