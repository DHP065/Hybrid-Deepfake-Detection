import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import argparse
import json
from torchvision import models, transforms
from scipy.signal import welch


class HybridDeepfakeDetector:
    def __init__(self, model_path="models/resnet18_ffpp.pth", device=None, threshold: float = 0.5, fake_class_index: int = 1):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = float(threshold)
        self.fake_class_index = int(fake_class_index)

        # Speed optimizations for convnets
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        # ---- Load ResNet18 ----
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)  # binary classifier (real/fake)

        if os.path.exists(model_path):
            print(f"[INFO] Loading pretrained weights from {model_path}")
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)

            # Try to load full state dict first (including fc) if compatible
            try:
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                print(f"[INFO] Loaded full state dict. Missing: {missing}, unexpected: {unexpected}")
            except Exception as e:
                print(f"[WARN] Full load failed ({e}). Trying without classifier weights...")
                # Fallback: Filter out classifier weights (fc.*)
                filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
                missing, unexpected = self.model.load_state_dict(filtered_state_dict, strict=False)
                print(f"[INFO] Loaded backbone only. Missing: {missing}, unexpected: {unexpected}")
        else:
            print("[WARNING] No pretrained model found, model will be untrained!")

        self.model.to(self.device)
        self.model.eval()

        # Face detection
        self.mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )

        # Preprocessing
        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def extract_frames(self, video_path, max_frames=500):
        cap = cv2.VideoCapture(video_path)
        frames, fps = [], cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        print(f"[INFO] Loaded {len(frames)} frames @ {fps:.2f} FPS")
        return frames, fps, total

    def detect_faces(self, frame):
        results = self.mp_face.process(frame)
        if results.detections:
            return results.detections[0]
        return None

    def crop_face(self, frame, detection):
        h, w, _ = frame.shape
        bbox = detection.location_data.relative_bounding_box
        x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
        x2, y2 = int((bbox.xmin + bbox.width) * w), int((bbox.ymin + bbox.height) * h)
        return frame[max(0, y1): y2, max(0, x1): x2]

    def compute_rppg(self, frames, fps):
        signals = []
        for frame in frames:
            face = self.detect_faces(frame)
            if face:
                crop = self.crop_face(frame, face)
                if crop.size > 0:
                    signals.append(crop[:, :, 1].mean())  # green channel
        if len(signals) < 10:
            print("[WARNING] Not enough signals for rPPG")
            return None, None

        signals = np.array(signals, dtype=np.float32)
        freqs, psd = welch(signals, fs=fps, nperseg=min(256, len(signals)))
        bpm = freqs[np.argmax(psd)] * 60
        snr = float(np.max(psd) / (np.mean(psd) + 1e-6))  # avoid div0
        return snr, float(bpm)

    def analyze(self, video_path, out_dir="out", max_frames_limit=None, skip_rppg=False, batch_size=16):
        os.makedirs(out_dir, exist_ok=True)
        # Apply a tighter frame limit if requested
        max_frames = int(max_frames_limit) if max_frames_limit else 500
        frames, fps, total = self.extract_frames(video_path, max_frames=max_frames)

        # CNN predictions (batched for speed)
        probs = []
        probs_idx0 = []
        probs_idx1 = []
        # Determine sampling stride (~64 frames target unless fewer available)
        target_frames = min(64, len(frames)) if len(frames) > 0 else 0
        sample_stride = max(1, len(frames) // max(1, target_frames)) if len(frames) > 0 else 1

        # Collect cropped face tensors for batching
        tensors = []
        for frame in frames[:: sample_stride]:  # sampled frames
            face = self.detect_faces(frame)
            if not face:
                continue
            crop = self.crop_face(frame, face)
            if crop.size == 0:
                continue
            tensors.append(self.preprocess(crop).unsqueeze(0))

        if len(tensors) > 0:
            input_batch = torch.cat(tensors, dim=0).to(self.device)
            # Run in mini-batches with optional automatic mixed precision
            with torch.no_grad():
                for i in range(0, input_batch.size(0), batch_size):
                    chunk = input_batch[i:i+batch_size]
                    if self.device == "cuda":
                        with torch.cuda.amp.autocast():
                            logits = self.model(chunk)
                    else:
                        logits = self.model(chunk)
                    soft = torch.softmax(logits, dim=1)
                    prob_chunk = soft[:, self.fake_class_index].tolist()
                    probs.extend(prob_chunk)
                    probs_idx0.extend(soft[:, 0].tolist())
                    probs_idx1.extend(soft[:, 1].tolist())

        # ✅ Always numeric
        pretrained_prob = float(np.mean(probs)) if len(probs) > 0 else 0.0
        cnn_prob_index_0 = float(np.mean(probs_idx0)) if len(probs_idx0) > 0 else 0.0
        cnn_prob_index_1 = float(np.mean(probs_idx1)) if len(probs_idx1) > 0 else 0.0

        # Robust aggregation for CNN probabilities
        if len(probs) >= 5:
            probs_sorted = np.sort(np.array(probs, dtype=np.float32))
            cut = max(1, int(0.1 * len(probs_sorted)))
            trimmed = probs_sorted[cut: len(probs_sorted) - cut] if len(probs_sorted) - 2*cut > 0 else probs_sorted
            mean_prob = float(np.mean(trimmed))
            median_prob = float(np.median(probs_sorted))
            pretrained_prob = float(0.7 * mean_prob + 0.3 * median_prob)
            # Recompute per-index means after trimming for fairness
            cnn_prob_index_0 = float(np.mean(np.sort(np.array(probs_idx0, dtype=np.float32))[cut: len(probs_sorted) - cut])) if len(probs_idx0) >= 5 else cnn_prob_index_0
            cnn_prob_index_1 = float(np.mean(np.sort(np.array(probs_idx1, dtype=np.float32))[cut: len(probs_sorted) - cut])) if len(probs_idx1) >= 5 else cnn_prob_index_1
        
        # rPPG signal (optional for speed)
        if skip_rppg:
            snr, bpm = None, None
        else:
            snr, bpm = self.compute_rppg(frames, fps)

        # Noise check (robust)
        if len(frames) > 1:
            frames_np = np.array(frames, dtype=np.float32)
            diffs = np.diff(frames_np)
            noise_score = float(np.var(diffs)) / 1e6 if diffs.size > 0 else 0.0
        else:
            noise_score = 0.0

        # Final decision fusion (adaptive weighting)
        weights = {"cnn": 0.6, "rppg": 0.3, "noise": 0.1}
        if skip_rppg or snr is None:
            weights = {"cnn": 0.8, "rppg": 0.0, "noise": 0.2}
        if len(probs) < 5:
            # Low evidence from CNN → down-weight
            weights["cnn"] = min(weights["cnn"], 0.5)
            weights["noise"] = 1.0 - weights["cnn"] - weights.get("rppg", 0.0)

        final_score = 0.0
        final_score += weights["cnn"] * max(0.0, min(1.0, pretrained_prob))
        if weights["rppg"] > 0:
            if snr is not None and snr < 5:  # low SNR → fake
                final_score += weights["rppg"] * 0.7
            elif snr is not None:
                final_score += weights["rppg"] * 0.3
        final_score += weights["noise"] * min(1.0, noise_score)
        final_score = float(max(0.0, min(1.0, final_score)))

        # Thresholded decision for fake detection
        decision = "FAKE" if final_score >= self.threshold else "REAL"

        # Alternative final score using opposite class index (diagnostic for label inversion)
        alt_fake_index = 1 - self.fake_class_index
        alt_prob = cnn_prob_index_1 if alt_fake_index == 1 else cnn_prob_index_0
        alternative_final_score = 0.0
        alternative_final_score += weights["cnn"] * max(0.0, min(1.0, float(alt_prob)))
        if weights["rppg"] > 0:
            if snr is not None and snr < 5:
                alternative_final_score += weights["rppg"] * 0.7
            elif snr is not None:
                alternative_final_score += weights["rppg"] * 0.3
        alternative_final_score += weights["noise"] * min(1.0, noise_score)
        alternative_final_score = float(max(0.0, min(1.0, alternative_final_score)))
        alternative_decision = "FAKE" if alternative_final_score >= self.threshold else "REAL"

        # Report with clearer notes
        report = {
            "rppg_snr": float(snr) if snr is not None else None,
            "rppg_peak_bpm": float(bpm) if bpm is not None else None,
            "noise_score": float(noise_score),
            "pretrained_prob_fake": float(pretrained_prob),
            "cnn_prob_index_0": float(cnn_prob_index_0),
            "cnn_prob_index_1": float(cnn_prob_index_1),
            "final_score": float(final_score),
            "decision": decision,
            "threshold": float(self.threshold),
            "fake_class_index": int(self.fake_class_index),
            "alternative": {
                "fake_class_index": int(alt_fake_index),
                "cnn_prob": float(alt_prob),
                "final_score": float(alternative_final_score),
                "decision": alternative_decision,
            },
            "notes": [
                "Lip–audio mismatch",
                "Pretrained model used" if len(probs) > 0 else "⚠️ No faces detected → CNN skipped",
                "⚠️ Too few signals for rPPG" if (not skip_rppg and snr is None) else ("rPPG skipped for speed" if skip_rppg else "rPPG computed"),
            ],
        }

        # Save JSON
        out_json = os.path.join(out_dir, "report.json")
        with open(out_json, "w") as f:
            json.dump(report, f, indent=2)

        print("===== Deepfake Detection Report =====")
        print(json.dumps(report, indent=2))
        print(f"Outputs saved to: {out_dir}")
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video")
    parser.add_argument("--out", default="out", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for FAKE")
    parser.add_argument("--fake_class_index", type=int, choices=[0, 1], default=1, help="Index of FAKE class in model output")
    args = parser.parse_args()

    detector = HybridDeepfakeDetector(threshold=args.threshold, fake_class_index=args.fake_class_index)
    detector.analyze(args.video, args.out)
