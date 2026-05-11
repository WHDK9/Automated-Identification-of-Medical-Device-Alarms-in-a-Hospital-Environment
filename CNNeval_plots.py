# CNNeval_plots.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from CNNdataset import AlarmDataset
from CNNmodel_enhanced import AlarmCNN_Enhanced
from CNNtrain_full import EnhancedSpectrogramGenerator

# ==================== Configuration ====================
EXCEL_PATH      = 'Generator/merged_dataset.xlsx'
IMAGE_DIR       = 'Generator/merged_spectrograms'
MODEL_PATH      = 'best_cnn_enhanced.pth'
TEST_IDX_PATH   = 'test_indices.npy'
NUM_CLASSES     = 10
THRESHOLD       = 0.5
BATCH_SIZE      = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

CLASS_NAMES = [f'alarm_{i}' for i in range(NUM_CLASSES)]


# ==================== Load Model & Run Inference ====================

def get_predictions():
    """Load model and run inference on the saved test split."""
    print(f"Loading model from: {MODEL_PATH}")
    model = AlarmCNN_Enhanced(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    test_idx    = np.load(TEST_IDX_PATH).tolist()
    full_ds     = AlarmDataset(EXCEL_PATH, IMAGE_DIR, transform=val_transform)
    test_subset = Subset(full_ds, test_idx)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    all_probs  = np.vstack(all_probs)   # [N, 10]
    all_labels = np.vstack(all_labels)  # [N, 10]
    all_preds  = (all_probs >= THRESHOLD).astype(int)

    print(f"Test samples: {len(all_labels)}")
    return all_probs, all_preds, all_labels


# ==================== Plot 1: Per-Class Confusion Matrices ====================

def plot_confusion_matrices(all_preds, all_labels, save_path='confusion_matrices.png'):
    """
    2×5 grid of binary confusion matrices, one per alarm class.
    Layout matches reference: rows = True (Neg/Pos), cols = Predicted (Neg/Pos).
    """
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        y_true = all_labels[:, i].astype(int)
        y_pred = all_preds[:, i].astype(int)

        # Build 2×2 matrix  [[TN, FP], [FN, TP]]
        TN = ((y_true == 0) & (y_pred == 0)).sum()
        FP = ((y_true == 0) & (y_pred == 1)).sum()
        FN = ((y_true == 1) & (y_pred == 0)).sum()
        TP = ((y_true == 1) & (y_pred == 1)).sum()
        cm = np.array([[TN, FP], [FN, TP]])

        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        plt.colorbar(im, ax=ax)

        ax.set_xticks([0, 1]);  ax.set_xticklabels(['Neg', 'Pos'])
        ax.set_yticks([0, 1]);  ax.set_yticklabels(['Neg', 'Pos'])
        ax.set_xlabel('Predicted');  ax.set_ylabel('True')
        ax.set_title(CLASS_NAMES[i])

        # Annotate cells
        for r in range(2):
            for c in range(2):
                val   = cm[r, c]
                color = 'white' if val > cm.max() / 2 else 'black'
                ax.text(c, r, str(val), ha='center', va='center',
                        fontsize=13, color=color, fontweight='bold')

    plt.suptitle('Per-Class Binary Confusion Matrices (Test Set)',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ==================== Plot 2: Performance Comparison Bar Chart ====================

def plot_performance_comparison(all_preds, all_labels,
                                save_path='performance_comparison.png'):
    """
    Grouped bar chart: Accuracy / Precision / Recall / F1 per class.
    """
    metrics = {m: [] for m in ['Accuracy', 'Precision', 'Recall', 'F1-Score']}

    for i in range(NUM_CLASSES):
        y_true = all_labels[:, i].astype(int)
        y_pred = all_preds[:, i].astype(int)

        TN = ((y_true == 0) & (y_pred == 0)).sum()
        FP = ((y_true == 0) & (y_pred == 1)).sum()
        FN = ((y_true == 1) & (y_pred == 0)).sum()
        TP = ((y_true == 1) & (y_pred == 1)).sum()
        N  = len(y_true)

        accuracy  = (TP + TN) / (N + 1e-9)
        precision = TP / (TP + FP + 1e-9)
        recall    = TP / (TP + FN + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)

        metrics['Accuracy'].append(accuracy * 100)
        metrics['Precision'].append(precision * 100)
        metrics['Recall'].append(recall * 100)
        metrics['F1-Score'].append(f1 * 100)

    x      = np.arange(NUM_CLASSES)
    width  = 0.2
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    fig, ax = plt.subplots(figsize=(16, 7))
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (idx - 1.5) * width
        ax.bar(x + offset, metrics[metric], width,
               label=metric, color=color, alpha=0.85, edgecolor='white')

    ax.set_xlabel('Alarm Class', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Performance Metrics by Class (Test Set)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ==================== Plot 3: Probability Distributions ====================

def plot_probability_distributions(all_probs, all_labels,
                                   save_path='probability_distributions.png'):
    """
    2×5 grid of probability histograms per class.
    Blue = negative samples, Red = positive samples,
    Green dashed line = decision threshold.
    """
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        probs  = all_probs[:, i]
        labels = all_labels[:, i].astype(int)

        neg_probs = probs[labels == 0]
        pos_probs = probs[labels == 1]

        bins = np.linspace(0, 1, 30)

        ax.hist(neg_probs, bins=bins, color='#4C72B0', alpha=0.7,
                label='Negative', edgecolor='white')
        ax.hist(pos_probs, bins=bins, color='#C44E52', alpha=0.7,
                label='Positive', edgecolor='white')
        ax.axvline(THRESHOLD, color='green', linestyle='--',
                   linewidth=1.5, label='Threshold')

        ax.set_title(CLASS_NAMES[i], fontsize=11)
        ax.set_xlabel('Probability', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7, loc='upper center')

    plt.suptitle('Predicted Probability Distributions by Class (Test Set)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ==================== Main ====================

if __name__ == '__main__':
    all_probs, all_preds, all_labels = get_predictions()

    plot_confusion_matrices(all_preds, all_labels)
    plot_performance_comparison(all_preds, all_labels)
    plot_probability_distributions(all_probs, all_labels)

    print("\nAll plots saved successfully.")
