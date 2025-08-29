#import tensorflow as tensorflow

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Conv1D
from tensorflow.keras.layers import Dense, Input, Concatenate, Masking, Dropout, BatchNormalization

from tensorflow.keras import regularizers 
from tensorflow.keras import optimizers

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model


from tensorflow.keras.metrics import TopKCategoricalAccuracy

from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
#import confusion matrix
#from sklearn.metrics import confusion_matrix

import pandas as pd

import tensorflow as tf
#import mediapipe as mp
import numpy as np
#import pickle
import os
import sys
from joblib import load
BATCH_SIZE = 32

# teacher student architecture 
# https://keras.io/examples/vision/knowledge_distillation/

"""

Ideas:
- LSTM Architecture then feed forward afterwards


"""

#remove 0081345 ... 6506-BAD


def wristNormalise(landmarks):
    wrist = landmarks[0]
    for lm in landmarks:
        lm[0] -= wrist[0]
        lm[1] -= wrist[1]
        lm[2] -= wrist[2]
    return landmarks

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

bestModelName = "RecentFinalArchitecture.h5"

lstmDropout = [0.16, 0.24, 0.32]
lstmRDropout = [0.1, 0.1, 0.15]
lstmL2 = [0.001, 0.002, 0.003]

denseDropout = [0.3, 0.25]
denseL2 = [0.003, 0.002]


train = pd.read_csv("trainss_164.csv")
val = pd.read_csv("valss_164.csv")

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
    try:
       data = load(file)
    except Exception as e:
        print(f"Error loading {file}: {e}", file=sys.stderr)
        continue
    
    trainLabels.append(row["Gloss"])

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
    vdata = load(file)
    

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
        sequence = wristNormalise(sequence)
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

#63 features for one hand
#Start 
leftIn = Input(shape=(None, 63), name = "LeftHand")
rightIn = Input(shape=(None, 63), name = "RightHand")

leftMask = Masking(mask_value=0.)(leftIn)
rightMask = Masking(mask_value=0.)(rightIn)

left1LSTM = LSTM(128, return_sequences=True, name="firstLeftLstm",
                  dropout = lstmDropout[0], recurrent_dropout = lstmRDropout[0],
                  kernel_regularizer=regularizers.l2(lstmL2[0])
                  )(leftMask)
right1LSTM = LSTM(128, return_sequences=True, name="firstRightLstm",
                   dropout = lstmDropout[0], recurrent_dropout = lstmRDropout[0],
                   kernel_regularizer=regularizers.l2(lstmL2[0])
                   )(rightMask)

left1LSTM = BatchNormalization()(left1LSTM)
right1LSTM = BatchNormalization()(right1LSTM)

left2LSTM = LSTM(128, return_sequences=True, name="secondLeftLstm",
                  dropout = lstmDropout[1], recurrent_dropout = lstmRDropout[1],
                  kernel_regularizer=regularizers.l2(lstmL2[1])
                  )(left1LSTM)
right2LSTM = LSTM(128, return_sequences=True, name="secondRightLstm",
                   dropout = lstmDropout[1], recurrent_dropout = lstmRDropout[1],
                   kernel_regularizer=regularizers.l2(lstmL2[1])
                   )(right1LSTM)

left2LSTM = BatchNormalization()(left2LSTM)
right2LSTM = BatchNormalization()(right2LSTM)

left3LSTM = LSTM(384, return_sequences=False, name="finalLeftLstm",
                  dropout = lstmDropout[2], recurrent_dropout = lstmRDropout[2],
                  kernel_regularizer=regularizers.l2(lstmL2[2])
                  )(left2LSTM)
right3LSTM = LSTM(384, return_sequences=False, name="finalRightLstm",
                  dropout = lstmDropout[2], recurrent_dropout = lstmRDropout[2],
                  kernel_regularizer=regularizers.l2(lstmL2[2])
                  )(right2LSTM)

combine = Concatenate(name="combined")([left3LSTM, right3LSTM])
X = Dense(384, activation='relu', kernel_regularizer = regularizers.l2(denseL2[0]))(combine)
X = Dropout(denseDropout[0])(X)
Y = Dense(256, activation='relu')(X)
Y = Dropout(denseDropout[1])(Y)
output = Dense(nclasses, activation='softmax', kernel_regularizer = regularizers.l2(denseL2[1]))(Y)

lossF = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)

# 2*384 = 

architecture = Model(inputs=[leftIn, rightIn], outputs=output)
architecture.compile(optimizer='adam',
                      loss=lossF,
                        metrics=['accuracy', TopKCategoricalAccuracy(k=3, name='top_3_accuracy')])
architecture.summary()

"""
Added extra dense layer due to 20% Val accuracy -> Top 3 val accuracy difference  
reduced recurrent regularisation - it was damaging 
"""

#End 
#checkpoint = ModelCheckpoint(bestModelName, monitor='val_accuracy',
 #                           verbose=1, save_best_only=True, mode='max')

#losscheckpoint = ModelCheckpoint("LossCheckpoint.h5", monitor='val_loss',
#                            verbose=1, save_best_only=True, mode='min')



#all_labels len = 164
label_map = {label: i for i, label in enumerate(all_labels)}
trainLabels = [label_map[label] for label in trainLabels]
valLabels = [label_map[label] for label in valLabels]


trainGenerator = Batch_Passer(trainLeft, trainRight, trainLabels)
valGenerator = Batch_Passer(valLeft, valRight, valLabels)

epoch_steps = len(trainLeft) // 32 
val_steps = len(valLeft) // 32

slowLR = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.7,
    patience = 10,
    verbose=1,  #print when reducing
    min_lr = 1e-6
)

#history = architecture.fit(
 #   trainGenerator, validation_data=valGenerator,
  #  steps_per_epoch = epoch_steps, epochs=360,
   # validation_steps = val_steps,
    #callbacks=[checkpoint, slowLR, losscheckpoint])



"""
Padded model converges at 84.5% validation, 71% test
Unpadded first run -> 90% validation, 67% test
Final first architecture = 91% validation, 80% test

Current architecture: 92.3% validation, 78.4% test


Randomly got 82%+ from a saved model 


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
        print(f"Error loading {file}: {e}", file=sys.stderr)
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

model = architecture
model.load_weights(bestModelName)

minlossModel = tf.keras.models.clone_model(model)
minlossModel.compile(optimizer='adam',
                      loss=lossF,
                        metrics=['accuracy', TopKCategoricalAccuracy(k=3, name='top_3_accuracy')])
minlossModel.load_weights("LossCheckpoint.h5")

# Initial evaluation
#loss, acc = evaluate_in_batches(model, testLeft, testRight, testLabels_numerical)
#print(f"Initial test accuracy: {acc*100:.2f}%")

import matplotlib.pyplot as plt
#moved graphing into a function so only the best model is saved
def postTrainingGenerations(history ):

# After training completes, add this code to visualize training progress
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig('training_history_164.png')
    plt.close()

    # Print summary statistics
    print("\nTraining Summary:")
    final_epochs = len(history.history['accuracy'])
    print(f"Trained for {final_epochs} epochs")
    print(f"Max validation accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
    print(f"Final training accuracy: {history.history['accuracy'][-1]*100:.2f}%")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
    train_val_gap = history.history['accuracy'][-1] - history.history['val_accuracy'][-1]
    print(f"Gap between train/val accuracy: {train_val_gap*100:.2f}%")



# Load and evaluate best model from training
loss1, acc1 = evaluate_in_batches(model, testLeft, testRight, testLabels_numerical)
print(f"Best val_acc model test accuracy: {acc1*100:.2f}%")
print(f"Best val_acc model loss: {loss1:.2f}")

loss2, acc2 = evaluate_in_batches(minlossModel, testLeft, testRight, testLabels_numerical)
print(f"Loss checkpoint model test accuracy: {acc2*100:.2f}%")
print(f"Loss checkpoint model loss: {loss2:.2f}")


if acc1 > acc2:
    print("Best model is the one with highest accuracy")
    bestModel = model
else:
    print("Best model is the one with lowest loss")
    bestModel = minlossModel
    model = minlossModel # switch 'model' to the one with the best test accuracy



def createVersionForTPU(model, name):
    convertor = tf.lite.TFLiteConverter.from_keras_model(model)
    
    convertor.optimizations = [tf.lite.Optimize.DEFAULT]
    convertor.target_spec.supported_types = [tf.float16]
    
    def representations():
        for start in range(0, min(100, len(testLeft)), 8):
            end = min(start + 8, len(testLeft))
            
            # Create small batches manually
            lhBatch = [testLeft[i] for i in range(start, end)]
            rhBatch = [testRight[i] for i in range(start, end)]
            
            # Use your existing preprocessing function
            leftPad = Preprocess_Batch(lhBatch)
            rightPad = Preprocess_Batch(rhBatch)
            
            # Only yield the inputs in the format your model expects
            yield [leftPad, rightPad]
                
    convertor.representative_dataset = representations
    try:
        tflite_model = convertor.convert()
        new = name.replace(".h5", "_tpu.tflite")
        with open(new, "wb") as f:
            f.write(tflite_model)
    except Exception as e:
        try:
            print("\n-------Error converting to TPU--------\n")
            print("Attemting basic conversion")
            basic = tf.lite.TFLiteConverter.from_keras_model(model)
            basicConv = basic.convert()
            with open(name.replace(".h5", "_basic.tflite"), "wb") as f:
                f.write(basicConv)
            print("Basic conversion saved")
        except Exception as e:
            print("Error converting to basic")
            print(e)




# Generate classification report
print("\n--- Generating Classification Report ---")

# Create directory for results if it doesn't exist
import os
os.makedirs("Results", exist_ok=True)

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
report_dict = classification_report(
    y_true, 
    y_pred,
    labels=np.arange(len(all_labels)),
    target_names=all_labels,
    digits=3,
    output_dict=True
)

report = classification_report(
    y_true, 
    y_pred,
    labels=np.arange(len(all_labels)),
    target_names=all_labels,
    digits=3,
)

class_metrics = {label: metrics for label, metrics in report_dict.items() if label in all_labels}
sorted_class_metrics = sorted(class_metrics.items(), key=lambda item: item[1]['f1-score'])



# Print the report
print("\n ------ Sorted Classification Report from Training: -----------")
print(f"{'Label':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")

for(label, metrics) in sorted_class_metrics:
    print(f"{label:<12} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1-score']:<10.3f} {metrics['support']:<8}")

#print(report)



# Save the report to a file

def saveNewModelsEval(report):
    with open("Results/classification_report.txt", "w") as f:
        f.write("Classification Report\n")
        f.write(f"Model: {bestModelName}\n")
        f.write(f"Test Accuracy: {acc1*100:.2f}%\n\n")
        f.write(report)
        
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
    with open("Results/confusion_pairs.txt", "w") as f:
        f.write("Most Common Confusion Pairs\n")
        f.write(f"Model: {bestModelName}\n\n")
        for true_label, pred_label, count in confusions:
            f.write(f"True: {true_label}, Predicted: {pred_label}, Count: {count}\n")

    print("Confusion pairs saved to Results/confusion_pairs.txt")



    # Create a visualization of the confusion matrix
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
    plt.savefig('Results/confusion_matrix.png', dpi=150)
    plt.close()

    print("Confusion matrix visualization saved to Results/confusion_matrix.png")
    print("Report saved to Results/classification_report.txt")



# Load and evaluate saved model
savedModelName = "Models/LastSize/LastDays.h5"

try:

    #see if the model exists
    if(not os.path.exists(savedModelName)):
        #
        print(" ---- Current Model Name does not exist ----")

        print("Plotting model history")
        #postTrainingGenerations(history)

        saveNewModelsEval(report)
        print("Saving model evaluation files")

        print("Saving model metadata")
        with open("BestModelDescription.txt", "w") as file: 
            file.write("Lstm dropout, regularisation and recurrent \n" + str(lstmDropout) + "\n" + str(lstmL2) + "\n" + str(lstmRDropout) + "\n")
            file.write("Dense dropout and regularisation \n" + str(denseDropout) + "\n" + str(denseL2) + "\n")
            file.write("Best model name: " + bestModelName + "\n")
            file.write("Best model accuracy: " + str(acc1) + "\n")
            file.write("Best model loss: " + str(loss1) + "\n")
            #file.write("Best model val accuracy: " + str(max(history.history['val_accuracy'])) + "\n")
        model.save(savedModelName)
        print("Saving Model model")

        #createVersionForTPU(model, savedModelName) # done locally 
        print("TPU version saved")
    else:
        model2 = load_model(savedModelName)
        loss2, acc2 = evaluate_in_batches(model2, testLeft, testRight, testLabels_numerical)
        print(f"Saved model accuracy: {acc2*100:.2f}%")

        if(acc1 > acc2):
            print("Better test accuracy, overwriting model")
            model.save(savedModelName, overwrite=True)
            print(".h5 saved")

            print("Overwriting evaluation files")

            saveNewModelsEval(report)


            print("Overwriting Model metadata txt")
            with open("BestModelDescription.txt", "w") as file: 
                file.write("Lstm dropout, regularisation and recurrent \n" + lstmDropout + "\n" + lstmL2 + "\n" + lstmRDropout + "\n")
                file.write("Dense dropout and regularisation \n" + denseDropout + "\n" + denseL2 + "\n")
                file.write("Best model name: " + bestModelName + "\n")
                file.write("Best model accuracy: " + str(acc1) + "\n")
                file.write("Best model loss: " + str(loss1) + "\n")
                #file.write("Best model val accuracy: " + str(max(history.history['val_accuracy'])) + "\n")

                
        else: 
            print("Saved model has better testing accuracy, not overwriting")
            print("Accuracy is", acc2)
            

except Exception as e:
    print("Error loading")
    print(e)
