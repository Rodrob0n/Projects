#import tensorflow as tensorflow

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Conv1D
from tensorflow.keras.layers import Dense, Input, Concatenate, Masking, Dropout, BatchNormalization

from tensorflow.keras import regularizers 
from tensorflow.keras import optimizers

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

from keras_tuner import RandomSearch, HyperModel

from keras_tuner import Hyperband

from tensorflow.keras.metrics import TopKCategoricalAccuracy

from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
#import confusion matrix
#from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
#import mediapipe as mp
import numpy as np
#import pickle
import os
from joblib import load
BATCH_SIZE = 32

# teacher student architecture 
# https://keras.io/examples/vision/knowledge_distillation/

"""

Ideas:
- LSTM Architecture then feed forward afterwards


"""

"""
Training for sign language detection, not amazing so far. Need to refine/Add more layers?

Switching to the unpadded dataset, saved by pickle.dump. Need to pad within batches



"""
#164 Labels
all_labels = ['LAST', 'TOMORROW', 'HORSE', 'THINK', 'IN', 'SLEEP', 'IMPORTANT', 'DIFFERENT', 'GOOD', 'SLOW', 'GRASS', 'FAST', 'FATHER', 'EMAIL', 'WEATHER', 'KITCHEN', 'SOME',
'INTERNET', 'GETTOGETHER', 'KNOW', 'NO', 'CAKE', 'OUT', 'WHY', 'TEXT', 'PLAY', 'WINTER', 'STREET', 'GAME', 'VEGETABLE', 'TRAIN', 'STOP', 'SPRING', 'EARTH', 'AFTER',
'SMALL', 'HAPPY', 'CAR', 'ANGRY', 'FIRE', 'NONE', 'ORANGE', 'SORRY', 'YESTERDAY', 'PARK', 'COME', 'COFFEE', 'LESS', 'MONTH', 'EASY', 'HOT', 'MOON', 'MOTHER', 'SIT',
'BASEBALL', 'NEXT', 'YEAR', 'WORK', 'METAL', 'TENNIS', 'BATHROOM', 'MORNING', 'LATER', 'TEA', 'FAMILY', 'RAIN', 'SELL', 'PICTURE', 'GO', 'CAMERA', 'BIRD', 'COUNTRY', 'DOOR', 'FOOTBALL',
'BOOK', 'NAME', 'SAD', 'TEACHER', 'HELLO', 'BEACH', 'WINDOW', 'LEARN', 'SUMMER', 'CLOSE', 'FRUIT',
'MORE', 'WHITE', 'APPLE', 'MANY', 'TRUE', 'UP', 'FRIEND', 'PLEASE', 'BROWN', 'FLOWER', 'GIRL', 'SILVER', 'SING', 'BIG', 'SISTER', 'HOME', 'CHAIR', 'SPORTS', 'COLOR',
'RESTAURANT', 'STAR', 'READ', 'UNDERSTAND', 'WATCH', 'MAYBE', 'DANCE', 'BUS', 'YELLOW', 'SEE', 'SCHOOL', 'STUDENT', 'RED', 'ANIMAL', 'PHONE', 'GREEN', 'SUN', 
'FIRST', 'WIND', 'COOK', 'BLUE', 'LOVE', 'YES', 'COLD', 'HELP', 'WATER', 'COOKIE', 'NEED', 'START', 'SHOES', 'BAD', 'OPEN', 'HOUSE', 'BOY', 'FISH', 'BEDROOM', 'NOW', 'TABLE', 'PEN', 'TREE', 'LONG', 'BEFORE',
'LISTEN', 'DAY', 'GOLD', 'NEW', 'WOOD', 'TIME', 'BROTHER', 'DOWN', 'ALL', 'BUY', 'OFFICE', 'WEEK', 'OLD', 'WRITE', 'FINISH', 'BIRTHDAY', 'COMPUTER', 'ICE']

ModelName = "TweakedFinalArchitecture.h5"
checkpointName = "TweakCheckpoint.h5"
original = "Models/LastSize/Best384.h5"

train = pd.read_csv("trainss_164.csv")
val = pd.read_csv("valss_164.csv")


"""lstmDropout = [0.1, 0.2, 0.3]
lstmRDropout = [0.1, 0.1, 0.15]
lstmL2 = [0.002, 0.003, 0.0035]
denseDropout = [0.3, 0.2]
denseL2 = [0.001, 0.002]"""
"""
Eighty+ test accuracy had
lstmDropout = [0.1, 0.2, 0.3]
lstmRDropout = [0.1, 0.1, 0.15]
lstmL2 = [0.0015, 0.002, 0.

denseDropout = [0.3, 0.2]
denseL2 = [0.001, 0.002]
"""






def format_data(data):
    num_frames =len(data)
    formatted = np.zeros((num_frames, 2, 21, 3))
    for i, frame in enumerate(data):
        for j in frame:
            side = 0 if j['side'] == 'Left' else 1
            landmarks = j['landmarks']
            formatted[i, side] = landmarks
    return formatted


#shape = (75,2,21,3)

trainLeft = []
trainRight = []
trainLabels = []
for idx, row in train.iterrows():
    file = f"164LabelLandmarks/{row['Video file']}"
    file = file.replace(".mp4", ".joblib")
    trainLabels.append(row["Gloss"])
    try:
        data = load(file)
    except Exception as e:
        print(e)
        continue
    data = format_data(data)
    left = data[:,0,:,:] #left hand
    right = data[:,1,:,:] #right hand
    
    trainLeft.append(left)
    trainRight.append(right)



valLeft = []
valRight = []
valLabels = []
for idx, row in val.iterrows():
    file = f"164LabelLandmarks/{row['Video file']}"
    file = file.replace(".mp4", ".joblib")
    try:
        vdata = load(file)
    except Exception as e:
        print(e)
        continue

    vdata = format_data(vdata)
    left = vdata[:,0,:,:] #left hand
    right = vdata[:,1,:,:] #right hand

    valLabels.append(row["Gloss"])
    valLeft.append(left)
    valRight.append(right)

    #for i in 
nclasses = len(all_labels)

#Pads batches dynamically and also does the reshaping
def Preprocess_Batch(batch_sequence):
    max_frames = max(len(sequence) for sequence in batch_sequence)
    batchSize = len(batch_sequence)
    padded = np.zeros((batchSize, max_frames, 63))

    for i, sequence in enumerate(batch_sequence):
        reshaped = sequence.reshape(len(sequence), -1) #flattening the 21x3
        padded[i, :len(sequence)] = reshaped # ignores the paddes frames; copies the actual recorded data within the 'new' frame
    return padded

#function to pass variable length batches to the model
def Batch_Passer(leftHand, rightHand, labels, batchSize=32):
    indexes = np.arange(len(leftHand))

    while True:
        batchDexes = np.random.choice(indexes, size=batchSize, replace=False)
        lhBatch = [leftHand[i] for i in batchDexes]
        rhBatch = [rightHand[i] for i in batchDexes]
        labelBatch = [labels[i] for i in batchDexes]

        leftPad = Preprocess_Batch(lhBatch)
        rightPad = Preprocess_Batch(rhBatch)

        labelBatch = to_categorical(labelBatch, num_classes=nclasses)

        yield [leftPad, rightPad], labelBatch
#


class CustomHyperModel(HyperModel):
    def __init__(self, pretrained=None):
        super().__init__()
        self.pretrained = pretrained

    def build(self, hp, pretrained_model=None):

        lstmDropout = [hp.Float('dropoutl1', 0.05,0.3,step=0.05, default = 0.1), hp.Float('dropoutl2', 0.1, 0.3, step=0.05, default = 0.1), hp.Float('dropoutl3', 0.15, 0.35, step=0.05, default = 0.15)]
        lstmRDropout = [0.1, 0.1, 0.15]
        lstmL2 = [hp.Float('l2lvl1', 5e-5, 5e-3, sampling="log"), hp.Float('l2lvl2', 5e-5, 5e-3, sampling="log"), hp.Float('l2lvl3', 1e-5, 6e-3,sampling="log")]
        denseDropout = [hp.Float("dropout_dense1", min_value=0.0, max_value=0.4, step=0.05, default=0.15), hp.Float("dropout_dense2", min_value=0.14, max_value=0.46, step=0.02, default=0.2)]
        denseL2 = [hp.Float('l2d1', 5e-5, 2e-4, sampling="log"),hp.Float('l2d2', 5e-5, 2e-4, sampling="log")]


        leftIn = Input(shape=(None, 63), name = "Left Hand")
        rightIn = Input(shape=(None, 63), name = "Right Hand")

        leftMask = Masking(mask_value=0.)(leftIn)
        rightMask = Masking(mask_value=0.)(rightIn)

        left1LSTM = LSTM(128, return_sequences=True, name="left_lstm1",
                        dropout = lstmDropout[0], recurrent_dropout = lstmRDropout[0],
                        kernel_regularizer=regularizers.l2(lstmL2[0])
                        )(leftMask)
        right1LSTM = LSTM(128, return_sequences=True, name="right_lstm1",
                        dropout = lstmDropout[0], recurrent_dropout = lstmRDropout[0],
                        kernel_regularizer=regularizers.l2(lstmL2[0])
                        )(rightMask)

        left1LSTM = BatchNormalization(trainable=False)(left1LSTM)
        right1LSTM = BatchNormalization(trainable=False)(right1LSTM)

        left2LSTM = LSTM(128, return_sequences=True, name="left_lstm2",
                        dropout = lstmDropout[1], recurrent_dropout = lstmRDropout[1],
                        kernel_regularizer=regularizers.l2(lstmL2[1])
                        )(left1LSTM)
        right2LSTM = LSTM(128, return_sequences=True, name="right_lstm2",
                        dropout = lstmDropout[1], recurrent_dropout = lstmRDropout[1],
                        kernel_regularizer=regularizers.l2(lstmL2[1])
                        )(right1LSTM)

        left2LSTM = BatchNormalization(trainable=False)(left2LSTM)
        right2LSTM = BatchNormalization(trainable=False)(right2LSTM)

        left3LSTM = LSTM(384, return_sequences=False, name="left_lstm3",
                        dropout = lstmDropout[2], recurrent_dropout = lstmRDropout[2],
                        kernel_regularizer=regularizers.l2(lstmL2[2])
                        )(left2LSTM)
        right3LSTM = LSTM(384, return_sequences=False, name="right_lstm3",
                        dropout = lstmDropout[2], recurrent_dropout = lstmRDropout[2],
                        kernel_regularizer=regularizers.l2(lstmL2[2])
                        )(right2LSTM)

        combine = Concatenate(name="combined")([left3LSTM, right3LSTM])
        X = Dense(384, activation='relu', kernel_regularizer = regularizers.l2(denseL2[0]))(combine)
        X = Dropout(denseDropout[0])(X)
        Y = Dense(256, activation='relu')(X)
        Y = Dropout(denseDropout[1])(Y)
        output = Dense(164, activation='softmax', kernel_regularizer = regularizers.l2(denseL2[1]))(Y)

        architecture = Model(inputs=[leftIn, rightIn], outputs=output)
        architecture.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                                metrics=['accuracy', TopKCategoricalAccuracy(k=3, name='top_3_accuracy')])
        architecture.summary()

        if pretrained_model:
            architecture.load_weights(pretrained_model)
            for layer in architecture.layers:
                if isinstance(layer, BatchNormalization):
                    layer.trainable = False
        
        
        
        return architecture 





#all_labels len = 164
label_map = {label: i for i, label in enumerate(all_labels)}
trainLabels = [label_map[label] for label in trainLabels]
valLabels = [label_map[label] for label in valLabels]


# Add this constant at the top of your file

trainGenerator = Batch_Passer(trainLeft, trainRight, trainLabels)
valGenerator = Batch_Passer(valLeft, valRight, valLabels)

epoch_steps = len(trainLeft) // 32 
val_steps = len(valLeft) // 32

slowLR = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.4,
    patience = 12,
    verbose=1,  #print when reducing
    min_lr = 1e-7
)


checkpoint_tune = ModelCheckpoint(
            'tuningValidationCheckpoint.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            save_weights_only=True,
            verbose=1
        )

architecture = CustomHyperModel(pretrained = original)

# Use Hyperband for hyperparameter tuning
tuner = Hyperband(
    architecture,
    objective='val_loss',
    max_epochs=100,
    factor=3,
    directory='tuning_dir',
    project_name='LSTM3yp',
    overwrite=True
)

tuner.search(
    trainGenerator,
    validation_data=valGenerator,
    steps_per_epoch=epoch_steps,
    validation_steps=val_steps,
    epochs=40,
    callbacks=[slowLR]
)

best_hypes = tuner.get_best_hyperparameters(1)[0]

model = tuner.hypermodel.build(best_hypes)

model.fit(
    trainGenerator,
    validation_data=valGenerator,
    steps_per_epoch=epoch_steps,
    validation_steps=val_steps,
    epochs=100,
    callbacks=[checkpoint_tune, slowLR]
)


print("Best hyperparameters:")
print(f"Dropout 1: {best_hypes.get('dropoutl1')}")
print(f"Dropout 2: {best_hypes.get('dropoutl2')}")
print(f"Dropout 3: {best_hypes.get('dropoutl3')}")
print(f"L2 Level 1: {best_hypes.get('l2lvl1')}")
print(f"L2 Level 2: {best_hypes.get('l2lvl2')}")
print(f"L2 Level 3: {best_hypes.get('l2lvl3')}")
print(f"Dense Dropout 1: {best_hypes.get('dropout_dense1')}")
print(f"Dense Dropout 2: {best_hypes.get('dropout_dense2')}")
print(f"Dense L2 1: {best_hypes.get('l2d1')}")
print(f"Dense L2 2: {best_hypes.get('l2d2')}")


"""
history = architecture.fit(
    trainGenerator, validation_data=valGenerator,
    steps_per_epoch = epoch_steps, epochs=100,
    validation_steps = val_steps,
    callbacks=[checkpoint_tune, slowLR])
"""

def TestBatchGenerator(leftHand, rightHand, labels, batchSize=32):
    total = len(leftHand)

    for start in range(0, total, batchSize):
        end_idx = min(start + batchSize, total)
        
        lhBatch = [leftHand[i] for i in range(start, end_idx)]
        rhBatch = [rightHand[i] for i in range(start, end_idx)]
        labelBatch = [labels[i] for i in range(start, end_idx)]
        
        leftPad = Preprocess_Batch(lhBatch)
        rightPad = Preprocess_Batch(rhBatch)
        labelBatch = to_categorical(labelBatch, num_classes=nclasses)
        
        yield [leftPad, rightPad], labelBatch



test = pd.read_csv("testss_164.csv")

# Load test data
testLeft = []
testRight = []
testLabels = []
for idx, row in test.iterrows():
    file = f"164LabelLandmarks/{row['Video file'].replace('.mp4', '.joblib')}"
    try:
        tdata = load(file)
    except Exception as e:
        print(e)
        continue
    tdata = format_data(tdata)

    left = tdata[:,0,:,:] #left hand
    right = tdata[:,1,:,:] #right hand

    testLabels.append(row["Gloss"])
    testLeft.append(left)
    testRight.append(right)
def evaluate_in_batches(model, leftHand, rightHand, labels, batch_size=32):
    """Evaluate model using batches without generator"""
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    for start in range(0, len(leftHand), batch_size):
        end = min(start + batch_size, len(leftHand))
        
        # Prepare batch
        lhBatch = [leftHand[i] for i in range(start, end)]
        rhBatch = [rightHand[i] for i in range(start, end)]
        labelBatch = [labels[i] for i in range(start, end)]
        
        # Preprocess
        leftPad = Preprocess_Batch(lhBatch)
        rightPad = Preprocess_Batch(rhBatch)
        labelBatch = to_categorical(labelBatch, num_classes=nclasses)
        
        # Evaluate batch
        metrics = model.test_on_batch(
            [leftPad, rightPad],
            labelBatch
        )
        total_loss += metrics[0]
        total_acc += metrics[1]
        num_batches += 1
    
    return total_loss/num_batches, total_acc/num_batches

# Replace your evaluation code with:
testLabels_numerical = [label_map[label] for label in testLabels]


# Function to get predictions for all test samples
def get_all_predictions(model, leftHand, rightHand):
    all_predictions = []
    
    for start in range(0, len(leftHand), 32):
        end = min(start + 32, len(leftHand))
        
        # Prepare batch
        lhBatch = [leftHand[i] for i in range(start, end)]
        rhBatch = [rightHand[i] for i in range(start, end)]
        
        # Preprocess
        leftPad = Preprocess_Batch(lhBatch)
        rightPad = Preprocess_Batch(rhBatch)
        
        # Get predictions
        batch_preds = model.predict([leftPad, rightPad], verbose=0)
        all_predictions.extend(batch_preds)
    
    return np.array(all_predictions)

# Generate predictions for your best model
print("Generating predictions...")
y_pred_prob = get_all_predictions(model, testLeft, testRight)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = testLabels_numerical

# Create classification report
report = classification_report(
    y_true, 
    y_pred,
    labels=np.arange(len(all_labels)),
    target_names=all_labels,
    digits=3
)

loss1, acc1 = evaluate_in_batches(model, testLeft, testRight, testLabels_numerical)
print(f"Best model accuracy: {acc1*100:.2f}%")
print(f"Best model loss: {loss1:.4f}")
# Print the report
print("\nClassification Report:")
print(report)

# Save the report to a file
with open("Results/tune_classification_report.txt", "w") as f:
    f.write("Classification Report\n")
    f.write(f"Model: {ModelName}\n")
    f.write(f"Test Accuracy: {acc1*100:.2f}%\n\n")
    f.write(report)

print("Report saved to Results/tune_classification_report.txt")

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Find most confused classes
print("\nTop 10 most confused classes:")
confusions = []
for i in range(len(cm)):
    for j in range(len(cm)):
        if i != j and cm[i, j] > 0:
            confusions.append((all_labels[i], all_labels[j], cm[i, j]))

# Sort by count and print top 10
confusions.sort(key=lambda x: x[2], reverse=True)
for true_label, pred_label, count in confusions[:10]:
    print(f"True: {true_label}, Predicted: {pred_label}, Count: {count}")

# Save confusion information
with open("Results/tunedconfusion_pairs.txt", "w") as f:
    f.write("Most Common Confusion Pairs\n")
    f.write(f"Model: {ModelName}\n\n")
    for true_label, pred_label, count in confusions:
        f.write(f"True: {true_label}, Predicted: {pred_label}, Count: {count}\n")

print("Confusion pairs saved to Results/tunedconfusion_pairs.txt")

# Optionally create a visualization of the confusion matrix
# Note: With 164 classes this might be hard to read
print("\nGenerating confusion matrix visualization...")
plt.figure(figsize=(40, 40))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(np.arange(len(all_labels)), all_labels, rotation=90, fontsize=8)
plt.yticks(np.arange(len(all_labels)), all_labels, fontsize=8)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('Results/tunedconfusion_matrix.png', dpi=150)
plt.close()

print("Confusion matrix visualization saved to Results/tunedconfusion_matrix.png")
