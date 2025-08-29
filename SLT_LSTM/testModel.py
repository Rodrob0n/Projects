from tensorflow.keras.models import load_model
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tensorflow.keras.utils import to_categorical

import random

def getLabels(prediction):
	out = []
	allLabels = [
    "APPLE", "BANANA", "ORANGE", "BOOK", "COMPUTER", "PHONE", "CAR", "BUS",
    "HOUSE", "DOOR", "WINDOW", "TABLE", "CHAIR", "BED", "FAMILY", "FRIEND",
    "SCHOOL", "WORK", "MONEY", "TIME", "DAY", "NIGHT", "MORNING", "EVENING",
    "HAPPY", "SAD", "ANGRY", "LOVE", "THANK YOU", "PLEASE", "SORRY", "HELLO",
    "GOODBYE", "COME", "GO", "RUN", "WALK", "SIT", "STAND", "READ", "WRITE",
    "SPEAK", "LISTEN", "SEE", "LOOK", "EAT", "DRINK", "COOK", "PLAY",
    "BUY", "SELL", "OPEN", "CLOSE", "START", "STOP", "BIG", "SMALL", "LONG",
    "SHORT", "HOT", "COLD", "NEW", "OLD", "FAST", "SLOW", "UP", "DOWN", "IN", "OUT"
	]
	for i in len(prediction):
		out.append(allLabels[np.argmax(prediction[i])])

	return out
		
	

def compareModels(p1, p2):
	model1 = load_model(p1)
	model2 = load_model(p2)
	sampleInt = 30
	test = pd.read_csv("testss.csv")
	videos = []
	truelabels = []

	for i in range(sampleInt):
		labelInt = random.randint(0, test.shape[0]-1)
		row = test.iloc[labelInt]
		video = cv2.VideoCapture(row["Video file"])
		label = row["Gloss"]
		videos.append(video)
		truelabels.append(label)
	#leftHand = []
	#rightHand = []
	mp_Hands = mp.solutions.hands
	hands = mp_Hands.Hands()
	
	for video in videos:
		videoData = []
		with hands as hands:
			while video.isOpened():
				ret, frame = video.read()
				if not ret:
					break
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				results = hands.process(frame)
				if results.multi_hand_landmarks and results.multi_handedness:
					for hand, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
						if handedness.classification[0].label == "Left":
							lh = np.array([[l.x, l.y, l.z] for l in hand.landmark])
							lh = lh.flatten()
							#leftHand.append(lh)
						else:
							rh = np.array([[r.x, r.y, r.z] for r in hand.landmark])
							rh = rh.flatten()
							#rightHand.append(rh)
						frameData = [lh, rh]
		videoData.append(frameData)

				
def getAccuracy(model):
	test = pd.read_csv("testss.csv")
	videos = []
	truelabels = []
	for idx, row in test.iterrows():
		video = cv2.VideoCapture(row["Video file"])
		label = row["Gloss"]
		videos.append(video)
		truelabels.append(label)
	leftHand = []
	rightHand = []
	mp_Hands = mp.solutions.hands
	hands = mp_Hands.Hands()
	with hands as hands:
		for video in videos:
			videoData = []
			while video.isOpened():
				ret, frame = video.read()
				if not ret:
					break
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				results = hands.process(frame)
				if results.multi_hand_landmarks and results.multi_handedness:
					for hand, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
						if handedness.classification[0].label == "Left":
							lh = np.array([[l.x, l.y, l.z] for l in hand.landmark])
							lh = lh.flatten()
							leftHand.append(lh)
						else:
							rh = np.array([[r.x, r.y, r.z] for r in hand.landmark])
							rh = rh.flatten()
							rightHand.append(rh)
						frameData = [lh, rh]
			videoData.append(frameData)
		leftHand = np.array(leftHand)
		rightHand = np.array(rightHand)
		leftHand = leftHand.reshape(1, -1)
		rightHand = rightHand.reshape(1, -1)
		prediction = model.predict([leftHand, rightHand])
		getLabels(prediction)
		print(getLabels(prediction==truelabels))
	

						
#input1 = load_model("Models/ThreeLSTM.keras")

#getAccuracy(input1)

model = load_model("RecentThreeLSTMBest.keras")
model.save("Models/CheckpointedThreeLSTM.h5")