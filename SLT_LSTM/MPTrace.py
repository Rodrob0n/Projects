import cv2 
import mediapipe as mp
import numpy as np 
import os 
from joblib import dump

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

source = "Filtered_Videos"
output = "164LabelLandmarks"

os.makedirs(output, exist_ok=True)

videos = [file for file in os.listdir(source) if file.endswith(".mp4")]
total_videos = len(videos)

print(f"Processing {total_videos} videos...")

skipCount = 0

for idx, video in enumerate(videos):
    path = os.path.join(source, video)
    
    output_path = os.path.join(output, video)
    output_path = output_path.replace(".mp4", ".joblib")


    #Uncomment to skip already processed videos
    
    if os.path.exists(output_path):
        skipCount +=1
        continue

    # Show progress periodically
    if idx % 10 == 0:
        print(f"Progress: {idx}/{total_videos} videos processed ({(idx/total_videos)*100:.1f}%)")
    
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error opening video {path}")
        continue

    frames_skel = []

    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.25,
        max_num_hands=2
        ) as Hands:
        frameid = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False

            results = Hands.process(img)
            
            frame_hands = []

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_data = {
                        'side': handedness.classification[0].label,
                        'landmarks': np.array([[l.x, l.y, l.z] for l in hand.landmark])
                    }
                    frame_hands.append(hand_data)

            if frame_hands:  # Only append frames where hands are detected
                frames_skel.append(frame_hands)

            frameid += 1
    cap.release()

    if frames_skel:  # Only save if there are frames with detected hands
                
        # Save directly as joblib
        dump(frames_skel, output_path)
    else:
        print(f"Warning: No hands detected in {video}")

print(f"Completed processing {total_videos} videos")
print(f" {skipCount} videos were already processed.")
print(f"Results saved to {output} directory")