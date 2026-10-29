# 🚀 Model Training Guide for Better Accuracy

This guide will help you train your deepfake detection model to achieve better accuracy using advanced techniques and more data.

## 📊 Current Model Analysis

Your current model:
- **Architecture**: ResNet18 trained on FaceForensics++
- **Current Performance**: ~65% confidence on sample video
- **Improvement Potential**: Can be significantly enhanced with proper training

## 🎯 Training Options

### Option 1: Quick Fine-tuning (Recommended for Beginners)
Use `quick_train.py` for immediate improvements:

```bash
# Step 1: Install additional dependencies
pip install seaborn tqdm

# Step 2: Prepare your data (see Data Preparation section)

# Step 3: Quick training (5-10 epochs)
python quick_train.py --epochs 10 --batch_size 16 --lr 1e-4
```

### Option 2: Advanced Training
Use `train_model.py` for comprehensive training with advanced techniques:

```bash
# Advanced training with ResNet50 and data augmentation
python train_model.py --architecture resnet50 --epochs 50 --use_mixup --use_focal_loss
```

### Option 3: Multiple Architecture Comparison
Train different architectures and compare:

```bash
# Train ResNet18
python train_model.py --architecture resnet18 --epochs 30

# Train ResNet50
python train_model.py --architecture resnet50 --epochs 30

# Train EfficientNet-B0
python train_model.py --architecture efficientnet_b0 --epochs 30
```

## 📁 Data Preparation

### Method 1: Prepare Your Own Data

1. **Collect Videos**:
   ```
   video_data/
     real/
       real_video1.mp4
       real_video2.mp4
     fake/
       fake_video1.mp4
       fake_video2.mp4
   ```

2. **Extract Faces**:
   ```bash
   python prepare_data.py --real_videos video_data/real --fake_videos video_data/fake --output_dir data
   ```

### Method 2: Download Public Datasets

```bash
# Get information about available datasets
python prepare_data.py --download_info
```

**Popular Datasets**:
1. **FaceForensics++** (Recommended)
   - 1,000 original videos + manipulated versions
   - Download: https://github.com/ondyari/FaceForensics

2. **Celeb-DF**
   - High-quality deepfake videos
   - Download: https://github.com/yuezunli/celeb-deepfakeforensics

3. **DFDC (Deepfake Detection Challenge)**
   - Large-scale dataset (100,000+ videos)
   - Download: https://ai.facebook.com/datasets/dfdc/

## 🏋️ Training Process

### Step 1: Data Preparation
```bash
# Extract faces from videos and create balanced dataset
python prepare_data.py \
  --real_videos path/to/real/videos \
  --fake_videos path/to/fake/videos \
  --output_dir data \
  --faces_per_video 30 \
  --balance
```

### Step 2: Start Training
```bash
# Quick training (for immediate results)
python quick_train.py --epochs 15 --batch_size 32

# OR Advanced training (for best results)
python train_model.py \
  --architecture resnet18 \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4 \
  --use_mixup \
  --optimizer adamw
```

### Step 3: Monitor Progress
- Training loss and validation accuracy are displayed in real-time
- Best model is automatically saved to `models/improved_model.pth`
- Training curves are plotted and saved

### Step 4: Evaluate Performance
```bash
# Evaluate the trained model
python evaluate_model.py \
  --model_path models/improved_model.pth \
  --test_data data/test \
  --output_dir evaluation_results
```

## 📈 Expected Accuracy Improvements

| Training Method | Expected Accuracy | Time Required |
|----------------|------------------|---------------|
| **Current Model** | 65-70% | - |
| **Quick Fine-tuning** | 75-80% | 30 minutes |
| **Advanced Training** | 85-90% | 2-4 hours |
| **Multi-Architecture** | 90-95% | 6-8 hours |

## 🔧 Advanced Techniques

### 1. Data Augmentation
- Random rotations, flips, color jittering
- MixUp and CutMix for better generalization
- Automatically applied in training scripts

### 2. Transfer Learning
- Start from ImageNet pretrained weights
- Fine-tune on deepfake-specific features
- Significantly faster convergence

### 3. Advanced Architectures
- **ResNet50**: Better feature extraction
- **EfficientNet**: State-of-the-art efficiency
- **Vision Transformer**: Attention-based learning

### 4. Loss Functions
- **Focal Loss**: Handle class imbalance
- **Label Smoothing**: Prevent overfitting
- **Weighted CrossEntropy**: Balance real/fake samples

### 5. Optimization Strategies
- **AdamW**: Better weight decay
- **Cosine Annealing**: Smooth learning rate decay
- **Early Stopping**: Prevent overfitting

## 🎯 Training Tips for Best Results

### 1. Data Quality
- **More Data = Better Results**: Aim for 10,000+ face images per class
- **Diverse Sources**: Mix different deepfake techniques
- **Balanced Dataset**: Equal real and fake samples

### 2. Training Strategy
- **Start Small**: Begin with 10-20 epochs for quick results
- **Gradual Improvement**: Increase epochs for better accuracy
- **Monitor Overfitting**: Watch validation vs training accuracy

### 3. Hardware Optimization
- **GPU Acceleration**: 10-20x faster training
- **Batch Size**: Increase if you have more GPU memory
- **Multiple Workers**: Speed up data loading

### 4. Model Selection
- **Start with ResNet18**: Good balance of speed and accuracy
- **Upgrade to ResNet50**: Better accuracy, slower training
- **Try EfficientNet**: Best efficiency

## 🔄 Updating Your Web App

After training, update your web app to use the improved model:

1. **Replace the model file**:
   ```bash
   cp models/improved_model.pth models/resnet18_ffpp.pth
   ```

2. **Restart the web app**:
   ```bash
   streamlit run app.py --server.port 8504
   ```

3. **Test with videos**: You should see improved accuracy!

## 📊 Monitoring Training

### Real-time Metrics
- **Training Loss**: Should decrease over time
- **Validation Accuracy**: Should increase over time
- **Learning Rate**: Automatically adjusted by scheduler

### Saved Outputs
- **Best Model**: `models/improved_model.pth`
- **Training History**: `training_history.json`
- **Evaluation Report**: `evaluation_results/evaluation_report.json`
- **Visualizations**: Confusion matrices, ROC curves

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**:
   ```bash
   # Reduce batch size
   python quick_train.py --batch_size 8
   ```

2. **No Training Data**:
   ```bash
   # Check data structure
   python prepare_data.py --download_info
   ```

3. **Poor Performance**:
   - Increase training epochs
   - Add more diverse training data
   - Try different architecture

4. **Overfitting**:
   - Reduce learning rate
   - Add more data augmentation
   - Use early stopping

### Performance Optimization

1. **Faster Training**:
   ```bash
   # Use smaller architecture
   python train_model.py --architecture resnet18 --batch_size 64
   ```

2. **Better Accuracy**:
   ```bash
   # Use larger architecture with advanced techniques
   python train_model.py --architecture resnet50 --use_mixup --epochs 100
   ```

## 🏆 Next Steps

1. **Start with Quick Training**: Use `quick_train.py` for immediate improvements
2. **Collect More Data**: Download public datasets for better results
3. **Advanced Training**: Use `train_model.py` for state-of-the-art performance
4. **Evaluate Results**: Use `evaluate_model.py` to measure improvements
5. **Deploy Improved Model**: Update your web app with the better model

## 📞 Need Help?

- Check the console output for detailed progress information
- Review the generated evaluation reports
- Experiment with different hyperparameters
- Consider ensemble methods for maximum accuracy

Happy training! 🚀
