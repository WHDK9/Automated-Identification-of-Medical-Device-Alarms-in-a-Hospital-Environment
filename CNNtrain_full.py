# CNNtrain_full.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import librosa
from PIL import Image
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, ConcatDataset
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from CNNdataset import AlarmDataset
from CNNmodel_enhanced import AlarmCNN_Enhanced
from CNNrule_checker import AlarmRuleChecker

# ==================== Configuration ====================
EXCEL_PATH          = 'Generator/merged_dataset.xlsx'
IMAGE_DIR           = 'Generator/merged_spectrograms'
NOISE_EXCEL_PATH    = 'Generator_Noise/noise_dataset.xlsx'       
NOISE_IMAGE_DIR     = 'Generator_Noise/shuffled_spectrograms'    
ORIGINAL_AUDIO_DIR  = r'Sounds/Alarms-20251113T232529Z-1-001/Alarms'  # 0.wav ~ 9.wav
MODEL_SAVE_PATH     = 'best_cnn_enhanced.pth'

BATCH_SIZE      = 32
LEARNING_RATE   = 0.0001
EPOCHS          = 15
PATIENCE        = 10        # Early stopping patience
RULE_WEIGHT     = 0.3       # Rule fusion weight at inference

TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15

SR      = 16000
N_FFT   = 1024
HOP_LEN = 512               # 50% overlap

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Data Augmentation ====================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==================== Enhanced Spectrogram Generator ====================
class EnhancedSpectrogramGenerator:
    def __init__(self, sr=SR, n_fft=N_FFT, hop_length=HOP_LEN, fmax=4000):
        self.sr         = sr
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.fmax       = fmax

    def _limit_freq(self, D):
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        return D[freqs <= self.fmax, :]

    def generate_stft(self, audio):
        D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        D = self._limit_freq(np.abs(D))
        return librosa.amplitude_to_db(D, ref=np.max)

    def generate_harmonic(self, audio):
        D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        H, _ = librosa.decompose.hpss(D)
        H = self._limit_freq(np.abs(H))
        return librosa.amplitude_to_db(H, ref=np.max)

    def generate_delta(self, stft_db):
        return librosa.feature.delta(stft_db)

    def _norm_to_uint8(self, x):
        x = x - x.min()
        if x.max() > 0:
            x = x / x.max()
        return (x * 255).astype(np.uint8)

    def audio_to_image(self, audio, target_size=(224, 224)):
        import cv2
        ch0 = self.generate_stft(audio)
        ch1 = self.generate_harmonic(audio)
        ch2 = self.generate_delta(ch0)
        h, w = target_size
        c0 = cv2.resize(self._norm_to_uint8(ch0), (w, h))
        c1 = cv2.resize(self._norm_to_uint8(ch1), (w, h))
        c2 = cv2.resize(self._norm_to_uint8(ch2), (w, h))
        return Image.fromarray(np.stack([c0, c1, c2], axis=2))

    def file_to_image(self, audio_path, target_size=(224, 224)):
        audio, _ = librosa.load(audio_path, sr=self.sr)
        return self.audio_to_image(audio, target_size)


# ==================== Noise Dataset (image-based) ====================
class NoiseImageDataset(torch.utils.data.Dataset):
    """
    Load noise spectrogram images from Generator_Noise/shuffled_spectrograms/.
    All labels: alarm_0..alarm_9 = 0, no_alarm = 1.
    Compatible with the same label format as AlarmDataset (num_classes columns).
    """
    def __init__(self, excel_path, image_dir, transform=None, num_alarm_classes=10):
        self.image_dir  = image_dir
        self.transform  = transform
        self.num_classes = num_alarm_classes  # Total label width = num_alarm_classes (no_alarm excluded from output)

        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Noise Excel not found: {excel_path}")

        df = pd.read_excel(excel_path)
        # Keep only rows where the spectrogram file actually exists
        df = df[df['Spectrogram_File'].apply(
            lambda f: os.path.exists(os.path.join(image_dir, f))
        )].reset_index(drop=True)

        self.filenames = df['Spectrogram_File'].tolist()

        # Label: no_alarm=1, all alarm_i=0  →  multi-label vector length num_alarm_classes
        # We match AlarmDataset convention: labels are alarm_0..alarm_{n-1} only (no_alarm not in vector)
        # So noise label = all zeros  (no alarm present)
        self.labels = torch.zeros(len(self.filenames), num_alarm_classes, dtype=torch.float32)

        print(f"NoiseImageDataset: {len(self.filenames)} samples loaded from {image_dir}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.filenames[idx])
        try:
            # Noise spectrograms are single-channel (RGB saved via matplotlib viridis)
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


# ==================== Original Audio Anchor Dataset ====================
class OriginalAudioDataset(torch.utils.data.Dataset):
    """
    Repeat each of the 10 original alarm audio files `repeat` times.
    Forces the model to strongly learn original alarm features.
    Label: one-hot, alarm_i -> index i = 1.
    """
    def __init__(self, audio_dir, generator, transform, num_classes=10, repeat=30):
        self.samples = []
        for i in range(num_classes):
            path = os.path.join(audio_dir, f'{i}.wav')
            if not os.path.exists(path):
                print(f"Warning: original audio not found: {path}")
                continue
            label = torch.zeros(num_classes, dtype=torch.float32)
            label[i] = 1.0
            for _ in range(repeat):
                self.samples.append((path, label, i))
        self.generator = generator
        self.transform  = transform
        print(f"OriginalAudioDataset: {len(self.samples)} samples "
              f"({num_classes} classes x {repeat} repeats)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, cls_id = self.samples[idx]
        try:
            image = self.generator.file_to_image(path)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            image = Image.new('RGB', (224, 224), color='white')
        if self.transform:
            image = self.transform(image)
        return image, label


# ==================== Focal Loss ====================
class FocalLoss(nn.Module):
    def __init__(self, pos_weight, alpha=0.25, gamma=2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.alpha      = alpha
        self.gamma      = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='none'
        )
        probs   = torch.sigmoid(inputs)
        p_t     = probs * targets + (1 - probs) * (1 - targets)
        focal_w = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * focal_w * bce).mean()


# ==================== Train / Validation Loop ====================
def run_epoch(model, loader, criterion, optimizer=None, training=True):
    model.train() if training else model.eval()
    total_loss          = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader,
                                   desc='Train' if training else 'Val',
                                   leave=False):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    epoch_loss = total_loss / len(loader)
    hamming_acc = (all_preds == all_labels).mean() * 100

    # Per-class recall
    recalls = []
    for i in range(all_labels.shape[1]):
        tp = ((all_labels[:, i] == 1) & (all_preds[:, i] == 1)).sum()
        fn = ((all_labels[:, i] == 1) & (all_preds[:, i] == 0)).sum()
        recalls.append(tp / (tp + fn + 1e-9))
    avg_recall = np.mean(recalls) * 100

    return epoch_loss, hamming_acc, avg_recall


# ==================== Main ====================
if __name__ == '__main__':
    print(f"Device: {device}")

    # ---------- 1. Spectrogram generator ----------
    generator = EnhancedSpectrogramGenerator()

    # ---------- 2. Load & split main alarm dataset ----------
    print("Loading main alarm dataset...")
    full_dataset = AlarmDataset(EXCEL_PATH, IMAGE_DIR, transform=None)
    num_classes  = full_dataset.num_classes

    indices = list(range(len(full_dataset)))
    train_idx, temp_idx = train_test_split(
        indices, test_size=(VAL_SPLIT + TEST_SPLIT), random_state=42, shuffle=True
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT),
        random_state=42
    )
    np.save('test_indices.npy', test_idx)

    assert not set(train_idx) & set(val_idx),  "Train/Val overlap detected!"
    assert not set(train_idx) & set(test_idx), "Train/Test overlap detected!"
    print(f"Main dataset  ->  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # ---------- 3. Build augmented subsets ----------
    train_ds_main = Subset(
        AlarmDataset(EXCEL_PATH, IMAGE_DIR, transform=train_transform), train_idx
    )
    val_ds = Subset(
        AlarmDataset(EXCEL_PATH, IMAGE_DIR, transform=val_transform), val_idx
    )

    # ---------- 4. Load noise dataset ----------
    print("Loading noise dataset...")
    noise_train_ds = NoiseImageDataset(
        excel_path      = NOISE_EXCEL_PATH,
        image_dir       = NOISE_IMAGE_DIR,
        transform       = train_transform,
        num_alarm_classes = num_classes
    )

    # Split noise dataset: 85% train / 15% val (mirror main split ratio)
    noise_indices = list(range(len(noise_train_ds)))
    noise_train_idx, noise_val_idx = train_test_split(
        noise_indices, test_size=0.15, random_state=42, shuffle=True
    )
    noise_train_subset = Subset(noise_train_ds, noise_train_idx)

    # Noise validation subset uses val_transform
    noise_val_ds = NoiseImageDataset(
        excel_path        = NOISE_EXCEL_PATH,
        image_dir         = NOISE_IMAGE_DIR,
        transform         = val_transform,
        num_alarm_classes = num_classes
    )
    noise_val_subset = Subset(noise_val_ds, noise_val_idx)

    print(f"Noise dataset ->  Train: {len(noise_train_subset)} | Val: {len(noise_val_subset)}")

    # ---------- 5. Original audio anchor dataset ----------
    original_ds = OriginalAudioDataset(
        audio_dir   = ORIGINAL_AUDIO_DIR,
        generator   = generator,
        transform   = train_transform,
        num_classes = num_classes,
        repeat      = 30
    )

    # ---------- 6. Merge training sets ----------
    # Main alarm samples + noise samples + original audio anchors
    train_dataset = ConcatDataset([train_ds_main, noise_train_subset, original_ds])
    val_dataset   = ConcatDataset([val_ds, noise_val_subset])
    print(f"Merged training set size : {len(train_dataset)}")
    print(f"Merged validation set size: {len(val_dataset)}")

    # ---------- 7. Compute positive-class weights (main train set only) ----------
    all_labels_np = np.vstack([
        full_dataset[i][1].numpy() for i in train_idx
    ])
    pos_counts = all_labels_np.sum(axis=0)
    neg_counts = len(all_labels_np) - pos_counts
    pos_weight = torch.tensor(
        neg_counts / (pos_counts + 1e-6), dtype=torch.float32
    ).to(device)

    print("Positive-class weights:")
    for i, w in enumerate(pos_weight.cpu().numpy()):
        print(f"   alarm_{i}: {w:.2f}")

    # ---------- 8. Weighted sampler (class balancing) ----------
    sample_weights  = []
    pos_weight_np   = pos_weight.cpu().numpy()

    # Main alarm train samples
    for i in train_idx:
        _, lbl = full_dataset[i]
        lbl_np = lbl.numpy()
        w = max(
            [pos_weight_np[j] for j in range(len(lbl_np)) if lbl_np[j] > 0.5],
            default=1.0
        )
        sample_weights.append(float(w))

    # Noise train samples: weight = 1.0 (all-zero label, treat as background)
    for _ in range(len(noise_train_subset)):
        sample_weights.append(1.0)

    # Original audio anchor samples: high weight to ensure frequent sampling
    for _, lbl, cls_id in original_ds.samples:
        sample_weights.append(float(pos_weight_np[cls_id] * 2.0))

    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ---------- 9. Model / Loss / Optimizer ----------
    print("Initializing model...")
    model = AlarmCNN_Enhanced(num_classes=num_classes).to(device)

    criterion = FocalLoss(pos_weight=pos_weight, alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # ---------- 10. Training loop ----------
    print(f"Starting training ({EPOCHS} epochs, early stopping patience={PATIENCE})...")
    print("=" * 70)

    best_val_recall = 0.0
    no_improve      = 0

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print("-" * 70)

        tr_loss, tr_acc, tr_rec = run_epoch(
            model, train_loader, criterion, optimizer, training=True
        )
        va_loss, va_acc, va_rec = run_epoch(
            model, val_loader, criterion, training=False
        )
        scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Train  Loss: {tr_loss:.4f}  Acc: {tr_acc:.2f}%  Recall: {tr_rec:.2f}%")
        print(f"Val    Loss: {va_loss:.4f}  Acc: {va_acc:.2f}%  Recall: {va_rec:.2f}%")
        print(f"LR: {lr_now:.7f}")

        if va_rec > best_val_recall:
            best_val_recall = va_rec
            no_improve      = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Best model saved (Val Recall: {va_rec:.2f}%)")
        else:
            no_improve += 1
            print(f"No improvement ({no_improve}/{PATIENCE})")
            if no_improve >= PATIENCE:
                print("Early stopping triggered!")
                break

    print("=" * 70)
    print(f"Training complete! Best validation recall: {best_val_recall:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
