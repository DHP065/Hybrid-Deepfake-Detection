# Hybrid Deepfake Detector
**Detect deepfakes using a hybrid approach that combines spatial CNN features with temporal rPPG and noise‑analysis signals.**  

[![GitHub license](https://img.shields.io/github/license/iconic2409/hybrid-deepfake-detector)](https://github.com/iconic2409/hybrid-deepfake-detector/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/iconic2409/hybrid-deepfake-detector?style=social)](https://github.com/iconic2409/hybrid-deepfake-detector/stargazers)

---  

## 📖 Overview

Deepfake videos manipulate facial appearance frame‑by‑frame, making detection challenging. 
Our **Hybrid Deepfake Detector** fuses three complementary signals:

| Signal | What it captures | Implementation |
|--------|------------------|----------------|
| **Spatial CNN** | Visual artefacts in individual frames (e.g., blending, over‑sharp edges) | `hybrid_deepfake_detector.py` (ResNet‑based backbone) |
| **rPPG (remote photoplethysmography)** | Physiological pulse‑rate patterns that should be consistent across a real video | `prepare_data.py` extracts rPPG; `evaluate_model.py` uses it as a feature |
| **Noise‑analysis** | Camera‑sensor noise statistics that differ for synthetic content | Integrated in `evaluate_model.py` |

The three streams are concatenated and fed into a lightweight classifier that outputs a **deepfake probability**.

---  

## 🗂️ Repository structure

```
deepfake_mini_project(final)/
├─ .gitignore                # auto‑generated (see below)
├─ app.py                    # entry‑point UI (Streamlit/FastAPI)
├─ hybrid_deepfake_detector.py
├─ evaluate_model.py
├─ prepare_data.py
├─ train_model.py
├─ quick_train.py
├─ requirements.txt
├─ README.md                 # <‑ this file
├─ TRAINING_GUIDE.md
├─ UI_README.md
├─ PROJECT_REPORT.md
├─ PROJECT_REPORT.pdf
├─ run_app.bat               # Windows shortcut to launch the UI
├─ models/                   # pre‑trained weights (see Model‑weight handling)
│   └─ *.pth                 # ≤100 MB each, otherwise LFS (see below)
├─ samples/                  # small demo videos/images (≤5 MB total)
└─ assets/                   # images used in this README
    └─ architecture.png
```

---  

## ⚙️ Installation (Windows)

1. **Clone the repo** (already done locally) or pull the remote after the first push.  
2. **Create a virtual environment**:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
```

3. **Install dependencies**:

```powershell
pip install -r requirements.txt
```

4. **Download model weights** (if not present):
   - The repository ships two weight files (`spatial_cnn.pth`, `fusion_classifier.pth`) that are each **< 100 MB** and are tracked directly.
   - If any weight exceeds 100 MB, it will be stored via **Git LFS** and a download script will fetch it automatically. _(Currently not needed.)_

5. **Run the application**:

```powershell
.\run_app.bat
```

or directly:

```powershell
python app.py
```

The UI opens in the default browser (Streamlit) and lets you load a video for inference.

---  

## 📚 Training & Evaluation

- **Training guide:** see `TRAINING_GUIDE.md`.  
- **Quick training** (for demo purposes) with default hyper‑parameters:

```powershell
python quick_train.py
```

- **Evaluation:**

```powershell
python evaluate_model.py --data-dir samples/
```

The script prints accuracy, AUC, and a confusion matrix.

---  

## 📸 Demo & Sample Assets

The `samples/` folder contains a few short public‑domain video clips (≈ 1 MB each). 
You can try them by selecting **`samples/sample_real.mp4`** or **`samples/sample_fake.mp4`** in the UI.

> **NOTE:** All sample media are small, share‑able, and released under the CC0 license.

---  

## 🏗️ Architecture diagram

```mermaid
flowchart TD
    A[Input video] --> B[Frame extractor]
    B --> C[Spatial CNN (ResNet‑50)]
    B --> D[Temporal rPPG extractor]
    B --> E[Noise‑analysis module]
    C --> F[Feature concat]
    D --> F
    E --> F
    F --> G[Classifier (MLP)]
    G --> H[Deepfake probability]
```

*(The diagram file `assets/architecture.png` is a higher‑resolution PNG version for GitHub.)*

---  

## 📦 Technology stack

| Layer | Library / Tool |
|-------|----------------|
| **Python** | 3.10 |
| **Deep learning** | PyTorch, torchvision |
| **Signal processing** | NumPy, SciPy |
| **Web UI** | Streamlit |
| **Packaging** | requirements.txt, optional Git LFS |
| **CI (optional)** | GitHub Actions (template ready) |

---  

## 🔒 Security & Privacy

- No secrets, tokens, or private data are stored in the repo.  
- All model weights are **public** and can be freely redistributed.  
- The `run_app.bat` script only launches the local UI; it does not contact external services.

---  

## 📌 Limitations & Future Work

- **Dataset size:** Currently trained on the FaceForensics++ and DFDC subsets; performance may vary on newer deepfake generation methods.  
- **Real‑time inference:** The pipeline processes video at ~ 5 fps on a CPU; GPU acceleration will be added.  
- **Model‑weight size:** Larger models (e.g., EfficientNet‑B7) could improve accuracy but would require Git LFS.  

Future enhancements planned:

1. GPU‑accelerated inference.  
2. Integrated benchmark suite (FAU‑CS, Celeb‑DF).  
3. Export to ONNX for cross‑platform deployment.

---  

## 📄 License

This project is released under the **MIT License**. See `LICENSE` for details.

---  

### 🎉 Ready to explore?

```powershell
git add .
git commit -m "feat: initial project publish with professional README"
git push -u origin main
```

Happy detecting! 🚀
