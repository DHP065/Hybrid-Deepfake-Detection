# 🔍 Hybrid Deepfake Detector Web UI

A user-friendly web interface for the hybrid deepfake detection system.

## 🚀 Quick Start

1. **Launch the UI:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to the displayed URL (usually `http://localhost:8501`)

3. **Upload a video** and click "Analyze Video"

## ✨ Features

### 🎯 Main Interface
- **Drag & Drop Video Upload**: Support for MP4, AVI, MOV, MKV files
- **Real-time Analysis**: Progress tracking with status updates
- **Instant Results**: Clear FAKE/REAL detection with confidence scores

### 📊 Detailed Analytics
- **Interactive Visualizations**: Plotly charts showing detection breakdown
- **Confidence Gauge**: Visual representation of deepfake probability
- **Technical Metrics**: 
  - CNN model predictions
  - Heart signal (rPPG) analysis
  - Video noise/artifact detection
- **Downloadable Reports**: JSON format with all analysis data

### 🛠️ Technical Details
- **Hybrid Detection**: Combines multiple AI techniques
- **Model Information**: Shows loaded model status
- **Sample Testing**: Built-in test with provided sample video

## 🎨 UI Components

1. **Upload Section**: Video file selection with preview
2. **Analysis Engine**: Real-time processing with progress tracking
3. **Results Dashboard**: 
   - Primary detection result (FAKE/REAL)
   - Confidence gauge chart
   - Detailed metrics breakdown
   - Interactive visualizations
4. **Technical Panel**: Expandable section with detection methodology

## 📈 Visualization Features

- **Bar Charts**: Component-wise analysis breakdown
- **Gauge Charts**: Overall confidence visualization
- **Metric Cards**: Individual analysis components
- **Color Coding**: Red for fake detection, green for real

## 🔧 Configuration

The UI automatically:
- Loads the pretrained model on first use
- Handles temporary file management
- Provides responsive design for different screen sizes
- Caches the AI model for faster subsequent analyses

## 💡 Usage Tips

1. **File Size**: Keep videos under 200MB for optimal performance
2. **Format**: MP4 files generally work best
3. **Quality**: Higher quality videos provide more accurate results
4. **Duration**: Shorter videos (30s-2min) process faster

## 🎬 Sample Analysis

Use the "Try with Sample Video" button to test the system with the included fake video sample.
