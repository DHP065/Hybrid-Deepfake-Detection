#!/usr/bin/env python3
"""
Data Preparation Script for Deepfake Detection Training

This script helps prepare training data from various sources:
1. Extract faces from videos
2. Organize data into train/val splits
3. Balance datasets
4. Create synthetic augmented data
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import json
import random
from tqdm import tqdm
import mediapipe as mp
import shutil
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceExtractor:
    """Extract faces from videos using MediaPipe"""
    
    def __init__(self, min_detection_confidence=0.5):
        self.mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=0, 
            min_detection_confidence=min_detection_confidence
        )
    
    def extract_face_from_frame(self, frame):
        """Extract face from a single frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face.process(rgb_frame)
        
        if results.detections:
            detection = results.detections[0]  # Take the first (most confident) detection
            h, w, _ = frame.shape
            bbox = detection.location_data.relative_bounding_box
            
            # Convert to absolute coordinates
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)
            
            # Add some padding
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Extract face
            face = frame[y1:y2, x1:x2]
            return face
        
        return None
    
    def extract_faces_from_video(self, video_path, max_faces=50, skip_frames=5):
        """Extract faces from a video"""
        cap = cv2.VideoCapture(video_path)
        faces = []
        frame_count = 0
        
        while len(faces) < max_faces:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for diversity
            if frame_count % skip_frames == 0:
                face = self.extract_face_from_frame(frame)
                if face is not None and face.shape[0] > 50 and face.shape[1] > 50:
                    # Resize to standard size
                    face_resized = cv2.resize(face, (224, 224))
                    faces.append(face_resized)
            
            frame_count += 1
        
        cap.release()
        return faces

class DataPreparer:
    """Prepare and organize training data"""
    
    def __init__(self, output_dir="data"):
        self.output_dir = Path(output_dir)
        self.face_extractor = FaceExtractor()
        
        # Create directory structure
        (self.output_dir / "train" / "real").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train" / "fake").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "val" / "real").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "val" / "fake").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "test" / "real").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "test" / "fake").mkdir(parents=True, exist_ok=True)
    
    def process_video_directory(self, video_dir, label, max_videos=None, faces_per_video=20):
        """Process all videos in a directory"""
        video_dir = Path(video_dir)
        video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mov"))
        
        if max_videos:
            video_files = video_files[:max_videos]
        
        all_faces = []
        label_name = "real" if label == 0 else "fake"
        
        logger.info(f"Processing {len(video_files)} {label_name} videos...")
        
        for video_file in tqdm(video_files, desc=f"Extracting {label_name} faces"):
            try:
                faces = self.face_extractor.extract_faces_from_video(
                    str(video_file), 
                    max_faces=faces_per_video
                )
                for i, face in enumerate(faces):
                    all_faces.append((face, label, f"{video_file.stem}_{i}"))
            except Exception as e:
                logger.warning(f"Error processing {video_file}: {e}")
        
        return all_faces
    
    def split_and_save_data(self, all_data, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """Split data into train/val/test and save"""
        # Separate by class
        real_data = [(face, filename) for face, label, filename in all_data if label == 0]
        fake_data = [(face, filename) for face, label, filename in all_data if label == 1]
        
        logger.info(f"Total real faces: {len(real_data)}")
        logger.info(f"Total fake faces: {len(fake_data)}")
        
        # Split each class separately to maintain balance
        for data, class_name in [(real_data, "real"), (fake_data, "fake")]:
            if len(data) == 0:
                continue
            
            # Split data
            train_data, temp_data = train_test_split(data, test_size=(1-train_ratio), random_state=42)
            val_data, test_data = train_test_split(temp_data, test_size=(test_ratio/(val_ratio+test_ratio)), random_state=42)
            
            # Save splits
            self.save_split(train_data, "train", class_name)
            self.save_split(val_data, "val", class_name)
            self.save_split(test_data, "test", class_name)
            
            logger.info(f"{class_name} - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    def save_split(self, data, split, class_name):
        """Save a data split to disk"""
        split_dir = self.output_dir / split / class_name
        
        for i, (face, filename) in enumerate(data):
            output_path = split_dir / f"{filename}.jpg"
            cv2.imwrite(str(output_path), face)
    
    def balance_dataset(self):
        """Balance the dataset by augmenting the minority class"""
        for split in ["train", "val"]:
            real_dir = self.output_dir / split / "real"
            fake_dir = self.output_dir / split / "fake"
            
            real_count = len(list(real_dir.glob("*.jpg")))
            fake_count = len(list(fake_dir.glob("*.jpg")))
            
            logger.info(f"{split} - Real: {real_count}, Fake: {fake_count}")
            
            # Balance by augmenting minority class
            if real_count > fake_count:
                self.augment_class(fake_dir, real_count - fake_count)
            elif fake_count > real_count:
                self.augment_class(real_dir, fake_count - real_count)
    
    def augment_class(self, class_dir, num_augmentations):
        """Augment a class by creating variations of existing images"""
        existing_images = list(class_dir.glob("*.jpg"))
        
        if len(existing_images) == 0:
            return
        
        logger.info(f"Augmenting {class_dir.name} class with {num_augmentations} samples")
        
        for i in range(num_augmentations):
            # Pick a random image to augment
            base_image_path = random.choice(existing_images)
            image = cv2.imread(str(base_image_path))
            
            # Apply random augmentation
            augmented = self.apply_augmentation(image)
            
            # Save augmented image
            output_path = class_dir / f"aug_{i}_{base_image_path.stem}.jpg"
            cv2.imwrite(str(output_path), augmented)
    
    def apply_augmentation(self, image):
        """Apply random augmentation to an image"""
        # Random horizontal flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random brightness/contrast
        alpha = random.uniform(0.8, 1.2)  # Contrast
        beta = random.randint(-20, 20)    # Brightness
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        # Random rotation
        angle = random.randint(-15, 15)
        h, w = image.shape[:2]
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        
        # Random noise
        if random.random() > 0.7:
            noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
            image = cv2.add(image, noise)
        
        return image
    
    def create_dataset_info(self):
        """Create a JSON file with dataset information"""
        info = {}
        
        for split in ["train", "val", "test"]:
            info[split] = {}
            for class_name in ["real", "fake"]:
                class_dir = self.output_dir / split / class_name
                count = len(list(class_dir.glob("*.jpg")))
                info[split][class_name] = count
        
        # Save info
        with open(self.output_dir / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)
        
        logger.info("Dataset info saved to dataset_info.json")
        return info

def download_sample_data():
    """Download sample deepfake datasets for training"""
    logger.info("To train your model effectively, you'll need deepfake datasets.")
    logger.info("Here are some popular public datasets you can use:")
    logger.info("")
    logger.info("1. FaceForensics++ (FF++)")
    logger.info("   - Download from: https://github.com/ondyari/FaceForensics")
    logger.info("   - Contains real and manipulated videos")
    logger.info("")
    logger.info("2. Celeb-DF")
    logger.info("   - Download from: https://github.com/yuezunli/celeb-deepfakeforensics")
    logger.info("   - High-quality deepfake videos")
    logger.info("")
    logger.info("3. DFDC (Deepfake Detection Challenge)")
    logger.info("   - Download from: https://ai.facebook.com/datasets/dfdc/")
    logger.info("   - Large-scale dataset with diverse manipulations")
    logger.info("")
    logger.info("After downloading, organize your data as:")
    logger.info("  video_data/")
    logger.info("    real/")
    logger.info("      video1.mp4")
    logger.info("      video2.mp4")
    logger.info("    fake/")
    logger.info("      fake_video1.mp4")
    logger.info("      fake_video2.mp4")

def main():
    parser = argparse.ArgumentParser(description="Prepare training data for deepfake detection")
    parser.add_argument("--real_videos", type=str, help="Directory containing real videos")
    parser.add_argument("--fake_videos", type=str, help="Directory containing fake videos")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for processed data")
    parser.add_argument("--max_videos", type=int, help="Maximum number of videos to process per class")
    parser.add_argument("--faces_per_video", type=int, default=20, help="Number of faces to extract per video")
    parser.add_argument("--download_info", action="store_true", help="Show information about downloading datasets")
    parser.add_argument("--balance", action="store_true", help="Balance the dataset after creation")
    
    args = parser.parse_args()
    
    if args.download_info:
        download_sample_data()
        return
    
    if not args.real_videos or not args.fake_videos:
        logger.error("Please provide --real_videos and --fake_videos directories")
        logger.info("Use --download_info to see how to obtain training data")
        return
    
    # Initialize data preparer
    preparer = DataPreparer(args.output_dir)
    
    # Process videos
    all_data = []
    
    # Process real videos
    if os.path.exists(args.real_videos):
        real_faces = preparer.process_video_directory(
            args.real_videos, 
            label=0, 
            max_videos=args.max_videos,
            faces_per_video=args.faces_per_video
        )
        all_data.extend(real_faces)
    
    # Process fake videos
    if os.path.exists(args.fake_videos):
        fake_faces = preparer.process_video_directory(
            args.fake_videos, 
            label=1, 
            max_videos=args.max_videos,
            faces_per_video=args.faces_per_video
        )
        all_data.extend(fake_faces)
    
    if len(all_data) == 0:
        logger.error("No data processed. Check your video directories.")
        return
    
    # Split and save data
    preparer.split_and_save_data(all_data)
    
    # Balance dataset if requested
    if args.balance:
        preparer.balance_dataset()
    
    # Create dataset info
    info = preparer.create_dataset_info()
    
    logger.info("Data preparation completed!")
    logger.info(f"Dataset summary: {json.dumps(info, indent=2)}")

if __name__ == "__main__":
    main()
