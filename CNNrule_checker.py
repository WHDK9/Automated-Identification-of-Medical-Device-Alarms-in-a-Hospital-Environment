# CNNrule_checker.py
import numpy as np
import librosa


class AlarmRuleChecker:
    """
    Rule-based auxiliary classifier using FFT acoustic analysis.
    Corrects model predictions when CNN confidence is low by applying
    physics-informed rules derived from each alarm's spectral signature.
    """

    def __init__(self, sr=16000, n_fft=1024, hop_length=512):
        self.sr         = sr
        self.n_fft      = n_fft
        self.hop_length = hop_length

    # ==================== Acoustic Feature Extraction ====================

    def get_freq_energy(self, audio, freq_low, freq_high):
        """
        Compute mean spectral energy within a specified frequency band.

        Args:
            audio     : np.ndarray - input waveform
            freq_low  : float      - lower bound of frequency band (Hz)
            freq_high : float      - upper bound of frequency band (Hz)

        Returns:
            float - mean magnitude within the band
        """
        D     = np.abs(librosa.stft(audio, n_fft=self.n_fft,
                                    hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        mask  = (freqs >= freq_low) & (freqs <= freq_high)
        return D[mask, :].mean()

    def get_fundamental_freq(self, audio):
        """
        Estimate the fundamental frequency (F0) via pYIN algorithm.

        Returns:
            float - median F0 in Hz; 0 if no voiced frames detected
        """
        f0, _, _ = librosa.pyin(audio, fmin=100, fmax=4000, sr=self.sr)
        f0 = f0[~np.isnan(f0)]
        return np.median(f0) if len(f0) > 0 else 0

    # ==================== Rule-Based Scoring ====================

    def check_rules(self, audio):
        """
        Compute per-class rule confidence scores in range [0, 1].

        Each alarm class is evaluated against a set of acoustic rules
        derived from FFT spectral analysis of the original alarm signals.

        Args:
            audio : np.ndarray - input waveform (mono, sr=self.sr)

        Returns:
            np.ndarray, shape [10] - rule confidence score per class
        """
        scores  = np.zeros(10)
        f0      = self.get_fundamental_freq(audio)
        e_total = self.get_freq_energy(audio, 0, 4000)

        # ── alarm_0: F0 ~492Hz, spectral notch at 2500-3000Hz ────────────
        if 450 < f0 < 540:
            scores[0] += 0.4
        e_2500_3000 = self.get_freq_energy(audio, 2500, 3000)
        if e_2500_3000 / (e_total + 1e-6) < 0.08:   # spectral depression
            scores[0] += 0.4

        # ── alarm_1: F0 ~962Hz, simultaneous peaks at 1000Hz & 2900Hz ───
        e_1000 = self.get_freq_energy(audio, 950,  1050)
        e_2900 = self.get_freq_energy(audio, 2800, 3000)
        if e_1000 > e_total * 0.15 and e_2900 > e_total * 0.15:
            scores[1] += 0.6
        if 900 < f0 < 1020:
            scores[1] += 0.3

        # ── alarm_2: F0 ~3000Hz, repeated pure tone at 3000Hz ────────────
        e_3000 = self.get_freq_energy(audio, 2900, 3100)
        if e_3000 / (e_total + 1e-6) > 0.4:
            scores[2] += 0.6
        if f0 > 2800:
            scores[2] += 0.3

        # ── alarm_3: F0 ~700Hz, harmonic series throughout signal ─────────
        if 650 < f0 < 750:
            scores[3] += 0.5
        # Verify harmonics at 700, 1400, 2100, 2800 Hz
        for harmonic_freq in [700, 1400, 2100, 2800]:
            e_h = self.get_freq_energy(audio,
                                       harmonic_freq - 50,
                                       harmonic_freq + 50)
            if e_h > e_total * 0.05:
                scores[3] += 0.1

        # ── alarm_4: F0 ~424Hz, 500Hz harmonic series, aperiodic ─────────
        if 380 < f0 < 470:
            scores[4] += 0.3
        e_500    = self.get_freq_energy(audio, 450,  550)
        e_1000_4 = self.get_freq_energy(audio, 950,  1050)
        e_1500   = self.get_freq_energy(audio, 1450, 1550)
        if e_500 > 0 and e_1000_4 > 0 and e_1500 > 0:
            scores[4] += 0.4

        # ── alarm_5: F0 ~670Hz, alternating pulses at 2500Hz & 1500Hz ────
        if 620 < f0 < 720:
            scores[5] += 0.3
        e_2500   = self.get_freq_energy(audio, 2400, 2600)
        e_1500_5 = self.get_freq_energy(audio, 1400, 1600)
        if e_2500 > e_total * 0.1 and e_1500_5 > e_total * 0.1:
            scores[5] += 0.4

        # ── alarm_6: F0 ~614Hz, high-freq tone (1500-3500Hz) alternating
        #            with low-freq broad pulse (100-500Hz) ─────────────────
        if 570 < f0 < 660:
            scores[6] += 0.3
        e_high = self.get_freq_energy(audio, 1500, 3500)
        e_low  = self.get_freq_energy(audio, 100,  500)
        if e_high > e_total * 0.3 and e_low > e_total * 0.1:
            scores[6] += 0.4

        # ── alarm_7: F0 ~503Hz, strong stable formant at 2000Hz ──────────
        if 460 < f0 < 550:
            scores[7] += 0.3
        e_2000 = self.get_freq_energy(audio, 1900, 2100)
        if e_2000 / (e_total + 1e-6) > 0.2:
            scores[7] += 0.5

        # ── alarm_8: F0 ~585Hz, dual spectral peaks 1000-1500Hz &
        #            3000-3500Hz ─────────────────────────────────────────
        if 540 < f0 < 630:
            scores[8] += 0.2
        e_band1 = self.get_freq_energy(audio, 1000, 1500)
        e_band2 = self.get_freq_energy(audio, 3000, 3500)
        if e_band1 > e_total * 0.2 and e_band2 > e_total * 0.15:
            scores[8] += 0.6

        # ── alarm_9: F0 ~160Hz, strongest high-order harmonic at 1500Hz ──
        if 130 < f0 < 190:
            scores[9] += 0.5
        e_1500_9 = self.get_freq_energy(audio, 1400, 1600)
        e_500_9  = self.get_freq_energy(audio, 450,  550)
        if e_1500_9 > e_500_9 and e_1500_9 > e_total * 0.15:
            scores[9] += 0.4

        return np.clip(scores, 0, 1)

    # ==================== Score Fusion ====================

    def fuse_with_model(self, model_probs, audio, rule_weight=0.3):
        """
        Fuse CNN model probabilities with rule-based scores.

        Formula:
            fused = (1 - rule_weight) * model_probs
                  +      rule_weight  * rule_scores

        Args:
            model_probs : np.ndarray - CNN sigmoid output, shape [num_classes]
            audio       : np.ndarray - raw waveform for rule evaluation
            rule_weight : float      - weight assigned to rule scores (0~1)
                                       default 0.3  →  CNN 70%, rules 30%

        Returns:
            np.ndarray, shape [num_classes] - fused confidence scores
        """
        rule_scores = self.check_rules(audio)
        fused       = (1 - rule_weight) * model_probs + rule_weight * rule_scores
        return fused
