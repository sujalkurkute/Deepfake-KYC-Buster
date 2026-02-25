import cv2
import os
from tqdm import tqdm
import random

# ============================
# 🔴 CHANGE THESE TWO PATHS
# ============================

REAL_VIDEO_PATH = r"C:\Users\Lenovo\Downloads\archive (2)\FaceForensics++_C23\original"
FAKE_VIDEO_PATH = r"C:\Users\Lenovo\Downloads\archive (2)\FaceForensics++_C23\Deepfakes"

# ============================

OUTPUT_REAL = "dataset/train/real"
OUTPUT_FAKE = "dataset/train/fake"

os.makedirs(OUTPUT_REAL, exist_ok=True)
os.makedirs(OUTPUT_FAKE, exist_ok=True)


def extract_frames(video_folder, output_folder, label, max_videos=120):
    videos = os.listdir(video_folder)
    random.shuffle(videos)

    count = 0

    for video in tqdm(videos[:max_videos]):
        video_path = os.path.join(video_folder, video)

        cap = cv2.VideoCapture(video_path)
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Save 1 frame every 10 frames
            if frame_id % 10 == 0:
                frame = cv2.resize(frame, (224, 224))
                filename = f"{label}_{count}.jpg"
                cv2.imwrite(os.path.join(output_folder, filename), frame)
                count += 1

            frame_id += 1

        cap.release()

    print(f"Finished extracting {label} images")


print("Extracting REAL videos...")
extract_frames(REAL_VIDEO_PATH, OUTPUT_REAL, "real")

print("Extracting FAKE videos...")
extract_frames(FAKE_VIDEO_PATH, OUTPUT_FAKE, "fake")

print("✅ DONE!")