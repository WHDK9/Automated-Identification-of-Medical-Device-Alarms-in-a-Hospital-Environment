"""
Medical Alarm Noise Sample Generator 
"""

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
NUM_ALARMS = 10  # Keep consistent

# Noise file paths
NOISE_FOLDER = r"Sounds/Alarms-20251113T232529Z-1-001/noise"
NOISE_FILES = [os.path.join(NOISE_FOLDER, f"b{i}.wav")
               for i in range(7, 60) if i != 46]  # b7-b59, skip b46

# ==================== Output Directory Setup ====================
BASE_OUTPUT_FOLDER = "Generator_Noise"
OUTPUT_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "output_spectrograms")
SHUFFLED_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "shuffled_spectrograms")
AUDIO_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "synthetic_audio")
EXCEL_FILE = os.path.join(BASE_OUTPUT_FOLDER, "noise_dataset.xlsx")

# Spectrogram parameters
N_FFT = 2048
HOP_LENGTH = 512
FMAX = 8000

# Data generation configuration
NUM_AUGMENTATIONS_PER_NOISE = 12  # Generate 12 variants per noise file

# ==================== Preprocessing ====================
print("\n" + "="*60)
print(" Medical Alarm Noise Sample Generator")
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

# ==================== Preload Noise Audio ====================
def preload_noise_samples():
    noise_data = {}
    print("Preloading noise audio files...")
    valid_count = 0

    for file in NOISE_FILES:
        if not os.path.exists(file):
            continue
        try:
            noise_audio, sr = librosa.load(file, sr=SAMPLING_RATE)
            noise_data[file] = noise_audio
            valid_count += 1
            print(f"  {os.path.basename(file)}: {len(noise_audio)/SAMPLING_RATE:.2f}s")
        except Exception as e:
            print(f"  Failed to load {file}: {e}")

    print(f"Successfully loaded {valid_count} noise files")
    return noise_data

noise_cache = preload_noise_samples()

# ==================== Data Augmentation Functions ====================
def apply_noise_augmentation(audio, aug_index):
    """Augmentation strategies for noise samples"""
    aug_audio = audio.copy()

    # Select different augmentation combinations based on index
    if aug_index % 4 == 0:
        # Volume change
        gain_db = random.choice([-9, -6, -3, 3, 6, 9])
        aug_audio = aug_audio * (10 ** (gain_db / 20))

    elif aug_index % 4 == 1:
        # Time stretch
        rate = random.uniform(0.85, 1.15)
        aug_audio = librosa.effects.time_stretch(aug_audio, rate=rate)

    elif aug_index % 4 == 2:
        # Pitch shift
        n_steps = random.choice([-2, -1, 1, 2])
        aug_audio = librosa.effects.pitch_shift(aug_audio, sr=SAMPLING_RATE, n_steps=n_steps)

    else:
        # Add slight white noise
        noise_level = random.uniform(0.003, 0.008)
        noise = np.random.normal(0, noise_level, len(aug_audio))
        aug_audio = aug_audio + noise

    # Randomly crop or loop to target length
    target_length = SAMPLING_RATE * DURATION

    if len(aug_audio) > target_length:
        # Random crop
        start = random.randint(0, len(aug_audio) - target_length)
        aug_audio = aug_audio[start:start + target_length]
    elif len(aug_audio) < target_length:
        # Loop to fill
        repeats = (target_length // len(aug_audio)) + 1
        aug_audio = np.tile(aug_audio, repeats)[:target_length]

    # Normalize
    max_val = np.max(np.abs(aug_audio))
    if max_val > 0:
        aug_audio = aug_audio / max_val * 0.95

    return aug_audio

# ==================== Spectrogram Generation ====================
def generate_spectrogram(audio, file_name, save_audio=False, audio_name=None):
    """Generate and save mel spectrogram"""
    if len(audio) < N_FFT:
        print(f"  Warning: {file_name} audio too short, skipping")
        return False

    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-6:
        print(f"  Warning: {file_name} audio energy too low, skipping")
        return False

    if save_audio and audio_name:
        audio_path = os.path.join(AUDIO_FOLDER, audio_name)
        sf.write(audio_path, audio, SAMPLING_RATE)

    try:
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLING_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=128,
            fmin=0,
            fmax=min(FMAX, SAMPLING_RATE // 2)
        )

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        if np.all(np.isnan(mel_spec_db)) or np.all(np.isinf(mel_spec_db)):
            print(f"  Warning: {file_name} spectrogram invalid, skipping")
            return False

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
        plt.title('Noise Mel Spectrogram')
        plt.tight_layout()

        output_path = os.path.join(OUTPUT_FOLDER, file_name)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        return True

    except Exception as e:
        print(f"  Failed to generate spectrogram {file_name}: {e}")
        plt.close()
        return False

# ==================== Dataset Generation ====================
def generate_noise_dataset():
    print(" " + "="*60)
    print(" Starting noise sample dataset generation")
    print("="*60)

    data = []
    spectrogram_files = []
    sample_index = 0

    print(f"Generating noise samples ({NUM_AUGMENTATIONS_PER_NOISE} variants per original noise file)")

    for noise_file, noise_audio in tqdm(noise_cache.items(), desc="Processing noise"):
        noise_name = os.path.basename(noise_file).replace('.wav', '')

        for aug_idx in range(NUM_AUGMENTATIONS_PER_NOISE):
            # Apply data augmentation
            augmented_audio = apply_noise_augmentation(noise_audio, aug_idx)

            spectrogram_file = f"noise_{sample_index:05d}.png"
            audio_file = f"noise_{sample_index:05d}.wav"

            # Save first 10 audio samples for verification
            save_audio = (sample_index < 10)

            if generate_spectrogram(augmented_audio, spectrogram_file, save_audio, audio_file):
                spectrogram_files.append(spectrogram_file)

                # Labels: all alarm columns = 0, no_alarm = 1
                alarm_presence = [0] * NUM_ALARMS + [1]
                data.append(alarm_presence)
                sample_index += 1

    print(f"Total generated: {len(spectrogram_files)} noise samples")

    # Create DataFrame
    print("Creating dataset annotation file...")
    columns = [f"alarm_{i}" for i in range(NUM_ALARMS)] + ["no_alarm"]
    df = pd.DataFrame(data, columns=columns)
    df["Original_Spectrogram_File"] = spectrogram_files

    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Rename files
    print("Renaming files...")
    new_filenames = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Renaming"):
        old_name = os.path.join(OUTPUT_FOLDER, row["Original_Spectrogram_File"])
        new_name = os.path.join(SHUFFLED_FOLDER, f"noise_{i:05d}.png")

        if os.path.exists(old_name):
            os.replace(old_name, new_name)
            new_filenames.append(f"noise_{i:05d}.png")
        else:
            new_filenames.append("")

    # Update DataFrame
    df.drop(columns=["Original_Spectrogram_File"], inplace=True)
    df.insert(0, "Spectrogram_File", new_filenames)
    df['Total_Alarms'] = 0  # All noise samples have 0 alarms

    # Save Excel
    df.to_excel(EXCEL_FILE, index=False)

    print(f"Noise dataset generation complete!")
    print(f"  Spectrograms saved to: {SHUFFLED_FOLDER}/")
    print(f"  Audio samples saved to: {AUDIO_FOLDER}/ (first 10)")
    print(f"  Annotation file: {EXCEL_FILE}")
    print(f"  Total samples: {len(df)}")
    print(f"  Original noise files: {len(noise_cache)}")
    print(f"  Average samples per noise file: {len(df) / len(noise_cache):.1f}")

    return df

# ==================== Dataset Validation ====================
def validate_dataset(df):
    print(" " + "="*60)
    print(" Dataset Validation")
    print("="*60)

    missing_files = []
    for idx, row in df.iterrows():
        file_path = os.path.join(SHUFFLED_FOLDER, row['Spectrogram_File'])
        if not os.path.exists(file_path):
            missing_files.append(row['Spectrogram_File'])

    if missing_files:
        print(f"Found {len(missing_files)} missing files")
        print(f"  First 5: {missing_files[:5]}")
    else:
        print("All spectrogram files present")

    # Verify all samples are labeled as no_alarm
    invalid_labels = df[df['no_alarm'] != 1]
    if len(invalid_labels) > 0:
        print(f"Found {len(invalid_labels)} label errors")
    else:
        print("All labels valid (no_alarm=1)")

    print(f"Dataset statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Feature dimensions: {len(df.columns) - 1}")
    print(f"  All samples are negative (no_alarm)")

    return len(missing_files) == 0 and len(invalid_labels) == 0

# ==================== Main Program ====================
def main():
    print("Configuration:")
    print(f"  Sample rate: {SAMPLING_RATE} Hz")
    print(f"  Sample duration: {DURATION} seconds")
    print(f"  FFT window: {N_FFT} ({N_FFT/SAMPLING_RATE*1000:.1f}ms)")
    print(f"  Hop length: {HOP_LENGTH} ({HOP_LENGTH/SAMPLING_RATE*1000:.1f}ms)")
    print(f"  Overlap: {(1 - HOP_LENGTH/N_FFT)*100:.0f}%")
    print(f"  Max frequency: {FMAX} Hz")
    print(f"  Augmentations per noise file: {NUM_AUGMENTATIONS_PER_NOISE}")

    if len(noise_cache) == 0:
        print("Error: No noise files found")
        print(f"Please check path: {NOISE_FOLDER}")
        return

    print(f"Found {len(noise_cache)} valid noise files")
    print(f"Expected to generate approximately {len(noise_cache) * NUM_AUGMENTATIONS_PER_NOISE} samples")

    response = input("Continue generating dataset? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled")
        return

    df = generate_noise_dataset()
    is_valid = validate_dataset(df)

    if is_valid:
        print("Dataset generated and validated successfully!")
        print("Tip: First 10 audio samples saved to Generator_Noise/synthetic_audio/")
        print("  Play these files to check audio quality")
    else:
        print("Dataset generation complete, but some issues were found - please review")

    print(" " + "="*60)
    print(" All tasks complete")
    print("="*60)

if __name__ == '__main__':
    main()
