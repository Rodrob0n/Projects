import mediapipe as mp
import cv2
import pandas as pd
import numpy as np
import os
import random
from collections import defaultdict

# Load dataset
ds = pd.read_csv('trainss_164.csv')

# Set up output directory
output_dir = "SkeletonVideos"
os.makedirs(output_dir, exist_ok=True)

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Group videos by gloss to select one per category
gloss_to_videos = defaultdict(list)
for _, row in ds.iterrows():
    gloss_to_videos[row['Gloss']].append(row['Video file'])

# Select one random video for each gloss
selected_videos = {}
for gloss, videos in gloss_to_videos.items():
    selected_videos[gloss] = random.choice(videos)

print(f"Selected {len(selected_videos)} videos, one for each unique sign")


lmdrawer = mp_drawing.DrawingSpec(
    color=(255, 255, 255),
    thickness=1,
    circle_radius = 1)

conndrawer = mp_drawing.DrawingSpec(
    color=(0, 255, 100),
    thickness=2,
    circle_radius = 1)


# Function to process a video
def process_video(video_file, gloss):
    # Input/output paths
    input_path = os.path.join("Filtered_Videos", video_file)
    output_path = os.path.join(output_dir, f"{gloss}_{os.path.basename(video_file)}")
    
    if not os.path.exists(input_path):
        print(f"Video not found: {input_path}")
        return False
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Failed to open {input_path}")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    slowed_fps = fps / 2  # Slow down the video to half speed 
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, slowed_fps, (width, height))
    
    # Process with Holistic
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.25) as holistic:
        
        frame_count = 0
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = holistic.process(image_rgb)
            
            # Create black background
            black_bg = np.zeros((height, width, 3), dtype=np.uint8)
            
            #draw face landmarks 
            """if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    black_bg,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=lmdrawer,
                    connection_drawing_spec = conndrawer)
"""
            # Draw pose landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    black_bg,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=lmdrawer,
                    connection_drawing_spec = conndrawer)
            
            # Draw hand landmarks
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    black_bg,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=lmdrawer,
                    connection_drawing_spec = conndrawer)
            
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    black_bg,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=lmdrawer,
                    connection_drawing_spec = conndrawer)
            

            if results.pose_landmarks and results.left_hand_landmarks:
                lwrist = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST]
                larmwrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST]

                x1 = int(lwrist.x * width)
                x2 = int(larmwrist.x * width)

                y1 = int(lwrist.y * height)
                y2 = int(larmwrist.y * height)
                cv2.line(black_bg, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if results.pose_landmarks and results.right_hand_landmarks:
                rwrist = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST]
                rarmwrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]

                x1 = int(rwrist.x * width)
                x2 = int(rarmwrist.x * width)

                y1 = int(rwrist.y * height)
                y2 = int(rarmwrist.y * height)
                cv2.line(black_bg, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add gloss label at the top
            cv2.putText(black_bg, f"Sign: {gloss}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add frame counter
            #frame_count += 1
            #cv2.putText(black_bg, f"Frame: {frame_count}", (10, 70), 
            #          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            
            # Write the frame
            out.write(black_bg)
    
    # Release resources
    cap.release()
    out.release()
    print(f"Processed {gloss}: {video_file}")
    return True

# Process each selected video
successful = 0
failed = 0

for gloss, video_file in selected_videos.items():
    print(f"Processing {gloss}: {video_file}")
    if process_video(video_file, gloss):
        successful += 1
    else:
        failed += 1

print(f"Processing complete! Successfully processed {successful} videos, failed on {failed}.")
print(f"Skeleton videos saved to {output_dir}/")