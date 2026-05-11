# CNNinference.py

import torch
import numpy as np
import librosa
from torchvision import transforms
from CNNmodel_enhanced import AlarmCNN_Enhanced
from CNNrule_checker import AlarmRuleChecker
from CNNtrain_full import EnhancedSpectrogramGenerator

# ==================== Configuration ====================
SR          = 16000
N_FFT       = 1024
HOP_LEN     = 512
NUM_CLASSES = 10
RULE_WEIGHT = 0.3
MODEL_PATH  = 'best_cnn_enhanced.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Inference Transform ====================
infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ==================== Prediction Function ====================
def predict(audio_path, threshold=0.65):
    """
    Run full inference pipeline on a single audio file.

    Pipeline:
        1. Load trained CNN model
        2. Generate 3-channel mel spectrogram
        3. CNN forward pass  →  per-class probabilities
        4. Acoustic rule checker  →  rule-based scores
        5. Weighted fusion  →  final decision

    Args:
        audio_path : str   - path to input .wav file
        threshold  : float - binary decision threshold (default 0.65)

    Returns:
        predictions : list[str] - detected alarm labels,
                                  e.g. ['alarm_1', 'alarm_5']
                                  or   ['no_alarm']
    """

    # ── Step 1: Load model ────────────────────────────────────────────────
    print(f"Loading model from: {MODEL_PATH}")
    model = AlarmCNN_Enhanced(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded successfully  (device: {device})")

    # ── Step 2: Generate 3-channel spectrogram ────────────────────────────
    print(f"Generating spectrogram for: {audio_path}")
    gen    = EnhancedSpectrogramGenerator(sr=SR, n_fft=N_FFT, hop_length=HOP_LEN)
    image  = gen.file_to_image(audio_path)           # PIL Image  RGB 224×224
    tensor = infer_transform(image).unsqueeze(0).to(device)   # [1, 3, 224, 224]

    # ── Step 3: CNN forward pass ──────────────────────────────────────────
    with torch.no_grad():
        logits      = model(tensor)                              # [1, num_classes]
        model_probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # [num_classes]

    # ── Step 4: Acoustic rule checker ─────────────────────────────────────
    print("Running acoustic rule checker...")
    audio, _ = librosa.load(audio_path, sr=SR)
    checker  = AlarmRuleChecker(sr=SR, n_fft=N_FFT, hop_length=HOP_LEN)

    # Rule scores: physics-based F0 / harmonic / periodicity analysis
    rule_scores = checker.check_rules(audio)   # [num_classes], range 0~1

    # ── Step 5: Weighted fusion ───────────────────────────────────────────
    # final_score = (1 - RULE_WEIGHT) * cnn_score + RULE_WEIGHT * rule_score
    #             =        0.7        * cnn_score +     0.3      * rule_score
    fused = checker.fuse_with_model(model_probs, audio, rule_weight=RULE_WEIGHT)

    # ── Step 6: Print results table ───────────────────────────────────────
    print(f"Audio file : {audio_path}")
    print(f"Threshold  : {threshold}  |  Rule weight: {RULE_WEIGHT}")
    print("=" * 65)
    print(f"{'Class':<12} {'CNN Prob':>10} {'Rule Score':>12} "
          f"{'Fused Score':>13} {'Decision':>8}")
    print("-" * 65)

    predictions = []
    for i in range(NUM_CLASSES):
        pred = fused[i] > threshold
        if pred:
            predictions.append(f"alarm_{i}")

        decision_marker = "O  ALARM" if pred else "X"
        print(f"alarm_{i:<6} {model_probs[i]:>10.3f} "
              f"{rule_scores[i]:>12.3f} "
              f"{fused[i]:>13.3f} "
              f"{decision_marker:>8}")

    print("-" * 65)

    # ── Step 7: Final decision summary ───────────────────────────────────
    if predictions:
        print(f"Detected alarms : {predictions}")
    else:
        print("Detected alarms : ['no_alarm']  — No alarm signal detected")

    print("=" * 65)

    return predictions if predictions else ["no_alarm"]


# ==================== Main Entry Point ====================
if __name__ == '__main__':
    import sys

    # Accept audio path from command line, fall back to default test file
    default_path = r'Sounds/Alarms-20251113T232529Z-1-001/Alarms/0.wav'
    audio_path   = sys.argv[1] if len(sys.argv) > 1 else default_path

    print("\n" + "=" * 65)
    print("  Medical Alarm Inference System")
    print("=" * 65)

    result = predict(audio_path)

    print(f"Final output : {result}")
