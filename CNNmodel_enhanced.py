# CNNmodel_enhanced.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyAttention(nn.Module):
    """
    Frequency-axis attention module.
    Directs model focus toward alarm-characteristic frequency bands:
    - Low band  (~160-500Hz):  fundamental frequency region
    - Mid band  (~700-1500Hz): dense harmonic region
    - High band (~2000-3500Hz): alarm signature region
    """
    def __init__(self, channels, freq_bins=56):
        super().__init__()
        self.freq_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d((freq_bins, 1)),
            nn.Flatten(1),
            nn.Linear(channels * freq_bins, 256),
            nn.ReLU(),
            nn.Linear(256, channels * freq_bins),
            nn.Sigmoid()
        )
        self.channels = channels
        self.freq_bins = freq_bins

    def forward(self, x):
        b, c, h, w = x.shape
        attn = self.freq_attn(x)
        attn = attn.view(b, c, self.freq_bins, 1)
        attn = F.interpolate(attn, size=(h, 1), mode='bilinear', align_corners=False)
        return x * attn.expand_as(x)


class TemporalAttention(nn.Module):
    """
    Temporal-axis attention module.
    Focuses on time positions of periodic events.
    """
    def __init__(self, channels):
        super().__init__()
        self.temp_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.temp_attn(x)
        return x * attn


class AlarmCNN_Enhanced(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Block 1: Capture low-level texture features
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Block 2: Capture harmonic patterns
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.freq_attn2 = FrequencyAttention(128)

        # Block 3: Capture high-level time-frequency structures
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.freq_attn3 = FrequencyAttention(256)
        self.temp_attn3 = TemporalAttention(256)

        # Block 4: Deeper feature extraction
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Global average pooling instead of flattening fully connected layers (reduces overfitting)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.freq_attn2(x)
        x = self.block3(x)
        x = self.freq_attn3(x)
        x = self.temp_attn3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
