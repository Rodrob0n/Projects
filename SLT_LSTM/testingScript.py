import cv2
import numpy as np
import os
import mediapipe as mp
import pandas as pd
import joblib

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils 

version = '164'

#depreciated
def testLandmarksOnVideo():

    # Paths to the video file and its corresponding landmark npy file.
    video_path = "SubsetVideos/30264522619849-PHONE.mp4"
    landmarks_path = "Landmark Data/30264522619849-PHONE.npy"

    # Load the saved landmarks (allow_pickle=True because we stored variable-length arrays).
    landmarks_data = np.load(landmarks_path, allow_pickle=True)
    outdir = "TestingOut"
    os.makedirs(outdir, exist_ok=True)
    
    # Open the video capture.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"{outdir}/landmarks_overlay2.mp4", fourcc, fps, (width, height))

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check if we have landmark data for this frame.
        h, w, _ = frame.shape
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame_index < len(landmarks_data):
            frame_landmarks = landmarks_data[frame_index]
            # frame_landmarks is expected to be a list (one element per detected hand).
            if frame_landmarks["lh"] is not None:
                for lm in frame_landmarks["lh"]:
                    x, y = int(lm[0] * w), int(lm[1] * h)
                    cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

            if frame_landmarks["rh"] is not None:
                for lm in frame_landmarks["rh"]:
                    x, y = int(lm[0] * w), int(lm[1] * h)
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        
        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()


#depreciated
def testLandmarkOutput():
    file = "LandmarkArrays/30264522619849-PHONE.npy"
    data = np.load(file, allow_pickle=True)
    print(data)
    print(data.shape)

#depreciated
def testLandmarkMeta():
    #print out the information of the dict
    file = "Landmark Data/30264522619849-PHONE.npy"
    data = np.load(file, allow_pickle=True)
    
    leftHand =[]
    rightHand = []

    file = np.load(file, allow_pickle=True)
    for frame in file:
        if frame["lh"] is None:
            print("None")
        if frame["lh"] is not None:
            print(frame["lh"])
            leftHand.append(frame["lh"])
        else:
            leftHand.append(np.zeros((21, 3)))

        if frame["rh"] is not None:
            rightHand.append(frame["rh"])
        else:
            rightHand.append(np.zeros((21, 3)))
    
    #print(np.array(leftHand).shape)
    #print(np.array(rightHand).shape)
    

def testNumberOfLabels():
    #iterate through trainss_70.csv
    print("70 Label Train Set Labels")
    print(len([
    "APPLE", "BANANA", "ORANGE", "BOOK", "COMPUTER", "PHONE", "CAR", "BUS",
    "HOUSE", "DOOR", "WINDOW", "TABLE", "CHAIR", "BED", "FAMILY", "FRIEND",
    "SCHOOL", "WORK", "MONEY", "TIME", "DAY", "NIGHT", "MORNING", "EVENING",
    "HAPPY", "SAD", "ANGRY", "LOVE", "THANK YOU", "PLEASE", "SORRY", "HELLO",
    "GOODBYE", "COME", "GO", "RUN", "WALK", "SIT", "STAND", "READ", "WRITE",
    "SPEAK", "LISTEN", "SEE", "LOOK", "EAT", "DRINK", "COOK", "PLAY", "WORK",
    "BUY", "SELL", "OPEN", "CLOSE", "START", "STOP", "BIG", "SMALL", "LONG",
    "SHORT", "HOT", "COLD", "NEW", "OLD", "FAST", "SLOW", "UP", "DOWN", "IN", "OUT"
]))
    print("164 Label Train Set Labels")
    print(len(['LAST', 'TOMORROW', 'HORSE', 'THINK', 'IN', 'SLEEP', 'IMPORTANT', 'DIFFERENT', 'GOOD', 'SLOW', 'GRASS', 'FAST', 'FATHER', 'EMAIL', 'WEATHER', 'KITCHEN', 'SOME',
'INTERNET', 'GETTOGETHER', 'KNOW', 'NO', 'CAKE', 'OUT', 'WHY', 'TEXT', 'PLAY', 'WINTER', 'STREET', 'GAME', 'VEGETABLE', 'TRAIN', 'STOP', 'SPRING', 'EARTH', 'AFTER',
'SMALL', 'HAPPY', 'CAR', 'ANGRY', 'FIRE', 'NONE', 'ORANGE', 'SORRY', 'YESTERDAY', 'PARK', 'COME', 'COFFEE', 'LESS', 'MONTH', 'EASY', 'HOT', 'MOON', 'MOTHER', 'SIT',
'BASEBALL', 'NEXT', 'YEAR', 'WORK', 'METAL', 'TENNIS', 'BATHROOM', 'MORNING', 'LATER', 'TEA', 'FAMILY', 'RAIN', 'SELL', 'PICTURE', 'GO', 'CAMERA', 'BIRD', 'COUNTRY', 'DOOR', 'FOOTBALL',
'BOOK', 'NAME', 'SAD', 'TEACHER', 'HELLO', 'BEACH', 'WINDOW', 'LEARN', 'SUMMER', 'CLOSE', 'FRUIT',
'MORE', 'WHITE', 'APPLE', 'MANY', 'TRUE', 'UP', 'FRIEND', 'PLEASE', 'BROWN', 'FLOWER', 'GIRL', 'SILVER', 'SING', 'BIG', 'SISTER', 'HOME', 'CHAIR', 'SPORTS', 'COLOR',
'RESTAURANT', 'STAR', 'READ', 'UNDERSTAND', 'WATCH', 'MAYBE', 'DANCE', 'BUS', 'YELLOW', 'SEE', 'SCHOOL', 'STUDENT', 'RED', 'ANIMAL', 'PHONE', 'GREEN', 'SUN', 
'FIRST', 'WIND', 'COOK', 'BLUE', 'LOVE', 'YES', 'COLD', 'HELP', 'WATER', 'COOKIE', 'NEED', 'START', 'SHOES', 'BAD', 'OPEN', 'HOUSE', 'BOY', 'FISH', 'BEDROOM', 'NOW', 'TABLE', 'PEN', 'TREE', 'LONG', 'BEFORE',
'LISTEN', 'DAY', 'GOLD', 'NEW', 'WOOD', 'TIME', 'BROTHER', 'DOWN', 'ALL', 'BUY', 'OFFICE', 'WEEK', 'OLD', 'WRITE', 'FINISH', 'BIRTHDAY', 'COMPUTER', 'ICE']))
    
def testMaxVideoLength():
    maxFrames = 0
    for file in os.listdir("LandmarkArrays"):
        data = np.load(f"LandmarkArrays/{file}", allow_pickle=True)
        maxFrames = max(maxFrames, data.shape[0])

    print("Max frames:", maxFrames)     
    print(maxFrames)

def testNumVideos():
    print(len(os.listdir("Filtered_Videos")))
    print(len(os.listdir("MPFrames_Joblib")))

def testSetSizes(version):
    print(f"Sizes of {version} label dataset splits")
    test = pd.read_csv(f"testss_{version}.csv")
    te = len(test)
    train = pd.read_csv(f"trainss_{version}.csv")
    tr =len(train)
    val = pd.read_csv(f"valss_{version}.csv")
    v = len(val)

    all = pd.merge(test, train, how="outer", on="Video file")
    all = pd.merge(all, val, how="outer", on="Video file")

    print(len(all))


    print("Train: ", tr) 
    print("Test: ", te)
    print("Val: ", v)
    print("Total: ", tr + te + v)

    train_test_overlap = set(train['Video file']).intersection(set(test['Video file']))
    print("Train and test overlap: ", len(train_test_overlap))

    print(f"\n--- SOURCE VIDEO VERIFICATION ---")
    filtered_dir = "Filtered_Videos"
    if not os.path.exists(filtered_dir):
        print(f"Error: {filtered_dir} directory not found!")
        return
    
    filtered_files = set(os.listdir(filtered_dir))
    all_dataset_files = set(all['Video file'])
    
    # Check for missing files
    missing_in_filtered = all_dataset_files - filtered_files
    print(f"Videos in dataset but missing from {filtered_dir}: {len(missing_in_filtered)}")
    if missing_in_filtered and len(missing_in_filtered) < 10:
        print("Examples:", list(missing_in_filtered)[:5])
    
    # Check for unused files
    unused_filtered = filtered_files - all_dataset_files  
    print(f"Videos in {filtered_dir} but not used in dataset: {len(unused_filtered)}")
    if unused_filtered and len(unused_filtered) < 10:
        print("Examples:", list(unused_filtered)[:5])
    
    print("Labels present in each")
    test = set(test["Gloss"].unique())
    train = set(train["Gloss"].unique())
    val = set(val["Gloss"].unique())
    print("Test:", len(test))
    print("Train:", len(train))
    print("Val:", len(val))

def testAvgVideoLength():
    totalFrames = 0
    dir = "MPFrames_Joblib"
    max = 0
    for file in os.listdir(dir):
        data = joblib.load(dir + "/" + file)
        #max = np.max(max, len(data))
        
        totalFrames += len(data)
        if(2*len(data) > 277):
            print(len(data))
            print(data)

    print("Average frames:", totalFrames / len(os.listdir(dir)))
    print(totalFrames / len(os.listdir(dir)))
    #print("Max frames:", max)


import pandas as pd
import numpy as np
from collections import Counter

def testForValuesInDirectory(version):
    if version == '164':
        directory = "164LabelLandmarks"
    else:
        directory = "MPFrames_Joblib"
    
    # Get all files in the directory
    files = os.listdir(directory)

    train = pd.read_csv(f"trainss_{version}.csv")
    test = pd.read_csv(f"testss_{version}.csv")
    val = pd.read_csv(f"valss_{version}.csv")

    all = pd.concat([train, test, val], ignore_index=True)
    # Remove duplicates
    all = all.drop_duplicates(subset=["Video file"])
    all = all.reset_index(drop=True)
    print(len(all))

    # separate glosses
    all_glosses = set(all["Gloss"].unique())
    print(len(all_glosses))

    # Initialize counter for total files found
    total_files_found = 0
    missing_glosses = []
    
    for gloss in all_glosses:
        # More precise matching - look for the gloss as a word in the filename
        # This prevents matching "IN" to words like "TRAIN" or "WINDOW"
        gloss_files = [f for f in files if f"-{gloss}." in f or f"-{gloss}-" in f]
        gloss_count = len(gloss_files)
        
        print(f"Gloss: {gloss}, Count: {gloss_count}")
        total_files_found += gloss_count
        
        if gloss_count == 0:
            missing_glosses.append(gloss)
    
    print(f"\nTotal files found for all glosses: {total_files_found}")
    print(f"Total unique files in directory: {len(files)}")
    
    if missing_glosses:
        print(f"\nMissing files for {len(missing_glosses)} glosses:")
        for gloss in missing_glosses:
            print(f"  - {gloss}")



def analyze_glosses(version):
    # Load all CSV files
    train_df = pd.read_csv(f"trainss_{version}.csv")
    test_df = pd.read_csv(f"testss_{version}.csv")
    val_df = pd.read_csv(f"valss_{version}.csv")
    
    # Expected labels from your model
    if(version == "70"):
        expected_labels = [
    "APPLE", "ORANGE", "BOOK", "COMPUTER", "PHONE", "CAR", "BUS",
    "HOUSE", "DOOR", "WINDOW", "TABLE", "CHAIR", "FAMILY", "FRIEND",
    "SCHOOL", "WORK", "TIME", "DAY", "MORNING", "PLEASE", "SORRY", "HELLO",
    "COME", "GO", "SIT", "READ", "WRITE", "LISTEN", "SEE", "COOK", "PLAY",
    "SELL", "OPEN", "CLOSE", "START", "STOP", "BIG", "SMALL", "LONG",
    "FAIL", "IMPROVE", "BETWEEN", "TOTAL", "GETTOGETHER", "WHY",
    "HAPPY", "SAD", "ANGRY", "LOVE", "BUY", "HOT", "COLD", "NEW", "OLD",
    "FAST", "SLOW", "UP", "DOWN", "IN", "OUT", "LIGHT", "GRAB", "YES", "NO",
    "EASY", "BOY", "GIRL", "COUGH", "MAYBE", "LAUGH"
    ]
    elif(version == "164"):
        expected_labels = ['LAST', 'TOMORROW', 'HORSE', 'THINK', 'IN', 'SLEEP', 'IMPORTANT', 'DIFFERENT', 'GOOD', 'SLOW', 'GRASS', 'FAST', 'FATHER', 'EMAIL', 'WEATHER', 'KITCHEN', 'SOME',
'INTERNET', 'GETTOGETHER', 'KNOW', 'NO', 'CAKE', 'OUT', 'WHY', 'TEXT', 'PLAY', 'WINTER', 'STREET', 'GAME', 'VEGETABLE', 'TRAIN', 'STOP', 'SPRING', 'EARTH', 'AFTER',
'SMALL', 'HAPPY', 'CAR', 'ANGRY', 'FIRE', 'NONE', 'ORANGE', 'SORRY', 'YESTERDAY', 'PARK', 'COME', 'COFFEE', 'LESS', 'MONTH', 'EASY', 'HOT', 'MOON', 'MOTHER', 'SIT',
'BASEBALL', 'NEXT', 'YEAR', 'WORK', 'METAL', 'TENNIS', 'BATHROOM', 'MORNING', 'LATER', 'TEA', 'FAMILY', 'RAIN', 'SELL', 'PICTURE', 'GO', 'CAMERA', 'BIRD', 'COUNTRY', 'DOOR', 'FOOTBALL',
'BOOK', 'NAME', 'SAD', 'TEACHER', 'HELLO', 'BEACH', 'WINDOW', 'LEARN', 'SUMMER', 'CLOSE', 'FRUIT',
'MORE', 'WHITE', 'APPLE', 'MANY', 'TRUE', 'UP', 'FRIEND', 'PLEASE', 'BROWN', 'FLOWER', 'GIRL', 'SILVER', 'SING', 'BIG', 'SISTER', 'HOME', 'CHAIR', 'SPORTS', 'COLOR',
'RESTAURANT', 'STAR', 'READ', 'UNDERSTAND', 'WATCH', 'MAYBE', 'DANCE', 'BUS', 'YELLOW', 'SEE', 'SCHOOL', 'STUDENT', 'RED', 'ANIMAL', 'PHONE', 'GREEN', 'SUN', 
'FIRST', 'WIND', 'COOK', 'BLUE', 'LOVE', 'YES', 'COLD', 'HELP', 'WATER', 'COOKIE', 'NEED', 'START', 'SHOES', 'BAD', 'OPEN', 'HOUSE', 'BOY', 'FISH', 'BEDROOM', 'NOW', 'TABLE', 'PEN', 'TREE', 'LONG', 'BEFORE',
'LISTEN', 'DAY', 'GOLD', 'NEW', 'WOOD', 'TIME', 'BROTHER', 'DOWN', 'ALL', 'BUY', 'OFFICE', 'WEEK', 'OLD', 'WRITE', 'FINISH', 'BIRTHDAY', 'COMPUTER', 'ICE']   
    
    # Get unique glosses from each dataset
    train_glosses = set(train_df["Gloss"].unique())
    test_glosses = set(test_df["Gloss"].unique())
    val_glosses = set(val_df["Gloss"].unique())
    
    # Count occurrences in each dataset
    train_counts = Counter(train_df["Gloss"])
    test_counts = Counter(test_df["Gloss"])
    val_counts = Counter(val_df["Gloss"])
    
    # All glosses across all datasets
    all_glosses = train_glosses.union(test_glosses).union(val_glosses)
    
    # Check for missing expected labels
    missing_expected = set(expected_labels) - all_glosses
    unexpected_found = all_glosses - set(expected_labels)
    
    # Print summary
    print(f"Total unique glosses across all datasets: {len(all_glosses)}")
    print(f"Expected number of labels: {len(expected_labels)}")
    
    print("\n--- Dataset Coverage ---")
    print(f"Training set: {len(train_glosses)} unique glosses")
    print(f"Testing set: {len(test_glosses)} unique glosses")
    print(f"Validation set: {len(val_glosses)} unique glosses")
    
    print("\n--- Missing from Datasets ---")
    print(f"Labels in expected list but not in any dataset: {len(missing_expected)}")
    if missing_expected:
        print("Missing labels:", sorted(missing_expected))
    
    print("\n--- Unexpected Labels ---")
    print(f"Labels found in datasets but not in expected list: {len(unexpected_found)}")
    if unexpected_found:
        print("Unexpected labels:", sorted(unexpected_found))
    
    print("\n--- Dataset Distribution ---")
    # Check if all glosses are in all datasets
    all_in_train = all_glosses.issubset(train_glosses)
    all_in_test = all_glosses.issubset(test_glosses)
    all_in_val = all_glosses.issubset(val_glosses)
    
    print(f"All glosses present in training set: {all_in_train}")
    print(f"All glosses present in testing set: {all_in_test}")
    print(f"All glosses present in validation set: {all_in_val}")
    
    # Report glosses missing from specific datasets
    print("\nMissing from training set:", sorted(all_glosses - train_glosses))
    print("\nMissing from testing set:", sorted(all_glosses - test_glosses))
    print("\nMissing from validation set:", sorted(all_glosses - val_glosses))
    
    # Print sample counts
    print("\n--- Sample Counts per Class ---")
    print("Format: GLOSS: train/test/val")
    
    for gloss in sorted(all_glosses):
        print(f"{gloss}: {train_counts.get(gloss, 0)}/{test_counts.get(gloss, 0)}/{val_counts.get(gloss, 0)}")
    

def testSetToDataMatches(version):
    train = pd.read_csv(f"trainss_{version}.csv")
    test = pd.read_csv(f"testss_{version}.csv")
    val = pd.read_csv(f"valss_{version}.csv")

    # Extract gloss sets from each split
    train_glosses = set(train["Gloss"])
    test_glosses = set(test["Gloss"])
    val_glosses = set(val["Gloss"])
    
    # Union of all glosses across splits
    all_glosses = train_glosses.union(test_glosses).union(val_glosses)
    print(f"Total unique glosses in dataset: {len(all_glosses)}")
    
    # Get all data files in the directory
    data_files = set(os.listdir("MPFrames_Joblib"))
    print(f"Total files in MPFrames_Joblib: {len(data_files)}")
    
    # Get all expected joblib files from the video filenames in the dataset
    all_videos = set()
    for _, row in pd.concat([train, test, val]).iterrows():
        video_file = row["Video file"].replace(".mp4", ".joblib")
        all_videos.add(video_file)
    
    print(f"Total unique videos in combined dataset: {len(all_videos)}")
    
    # Check files that are in directory but not in dataset
    extra_files = data_files - all_videos
    print(f"Files in directory but not in dataset: {len(extra_files)}")
    if extra_files and len(extra_files) < 10:
        print("Examples:", list(extra_files)[:5])
    
    # Check files that are in dataset but not in directory
    missing_files = all_videos - data_files
    print(f"Files in dataset but missing from directory: {len(missing_files)}")
    if missing_files and len(missing_files) < 10:
        print("Examples:", list(missing_files)[:5])
    
    # Check gloss distribution across splits
    print("\n--- Gloss Distribution ---")
    print(f"Glosses in train set: {len(train_glosses)}")
    print(f"Glosses in test set: {len(test_glosses)}")
    print(f"Glosses in val set: {len(val_glosses)}")
    
    return all_glosses, all_videos, data_files


def compareFilteredToJoblib():
    print("Comparing Filtered_Videos to MPFrames_Joblib...")
    
    # Check if directories exist
    if not os.path.exists("Filtered_Videos"):
        print("Error: Filtered_Videos directory not found!")
        return
        
    if not os.path.exists("MPFrames_Joblib"):
        print("Error: MPFrames_Joblib directory not found!")
        return
    
    # Get lists of files in both directories
    filtered_files = set(os.listdir("Filtered_Videos"))
    joblib_files = set(os.listdir("MPFrames_Joblib"))
    
    # Count files
    print(f"Total files in Filtered_Videos: {len(filtered_files)}")
    print(f"Total files in MPFrames_Joblib: {len(joblib_files)}")
    
    # Convert mp4 filenames to expected joblib filenames
    expected_joblib_files = {f.replace('.mp4', '.joblib') for f in filtered_files}
    
    # Check what's missing
    missing_joblibbed = expected_joblib_files - joblib_files
    print(f"Files in Filtered_Videos but missing from MPFrames_Joblib: {len(missing_joblibbed)}")
    if missing_joblibbed and len(missing_joblibbed) < 10:
        print("Examples:", list(missing_joblibbed)[:5])
    
    # Check for extra files
    extra_joblib = joblib_files - expected_joblib_files
    print(f"Files in MPFrames_Joblib but not from Filtered_Videos: {len(extra_joblib)}")
    if extra_joblib and len(extra_joblib) < 10:
        print("Examples:", list(extra_joblib)[:5])
    
    # Calculate percentage of videos that have been processed
    if len(filtered_files) > 0:
        completion_percentage = (len(expected_joblib_files) - len(missing_joblibbed)) / len(filtered_files) * 100
        print(f"Processed completion: {completion_percentage:.2f}%")
    
    return filtered_files, joblib_files, missing_joblibbed


def testForSpecificEntries():
    entries = ['GETTOGETHER']
    train = pd.read_csv("trainss_164.csv")
    test = pd.read_csv("testss_164.csv")
    val = pd.read_csv("valss_164.csv")

    # Extract gloss sets from each split
    train_vids = train[(train["Gloss"] == entries[0])]['Video file'].tolist()
    test_vids = test[(test["Gloss"] == entries[0])]['Video file'].tolist()
    val_vids = val[(val["Gloss"] == entries[0])]['Video file'].tolist()


    vdirectory = os.listdir("Filtered_Videos")
    ldirectory = os.listdir("164LabelLandmarks")

    print("Function will print when files are not present ")
    
    # Check for each entry in the list
    for entry in train_vids:
        entrylm = entry.replace(".mp4", ".joblib")
        if entrylm not in ldirectory:
            print(f"{entry} not found in {ldirectory}")
        if entry not in vdirectory:
            print(f"{entry} not found in {vdirectory}")
    for entry in test_vids:
        entrylm = entry.replace(".mp4", ".joblib")

        if entry not in vdirectory:
            print(f"{entry} not found in {vdirectory}")
        if entrylm not in ldirectory:
            print(f"{entry} not found in {ldirectory}")

    for entry in val_vids:
        entrylm = entry.replace(".mp4", ".joblib")

        if entry not in vdirectory:
            print(f"{entry} not found in {vdirectory}")
        if entrylm not in ldirectory:
            print(f"{entry} not found in {ldirectory}")
        
def testLabelDistribution(version):
    # Load the CSV files
    train_df = pd.read_csv(f"trainss_{version}.csv")
    test_df = pd.read_csv(f"testss_{version}.csv")
    val_df = pd.read_csv(f"valss_{version}.csv")

    # Count occurrences of each label in each set
    train_counts = train_df['Gloss'].value_counts()
    test_counts = test_df['Gloss'].value_counts()
    val_counts = val_df['Gloss'].value_counts()

    # Print the counts
    print("Train Set Label Distribution:")
    print(train_counts)
    print("\nTest Set Label Distribution:")
    print(test_counts)
    print("\nValidation Set Label Distribution:")
    print(val_counts)
        




#analyze_glosses(version)

#testLandmarksOnVideo()

#testLandmarkOutput()

#testNumberOfLabels()

#testLandmarkMeta()

#testMaxVideoLength()

#testNumVideos()

#testSetSizes(version)

#testAvgVideoLength()

#testForValuesInDirectory(version)

#testForSpecificEntries()

#testSetToDataMatches()

#compareFilteredToJoblib()

testLabelDistribution(164)