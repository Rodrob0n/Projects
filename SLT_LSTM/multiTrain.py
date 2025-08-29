from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Conv1D
from tensorflow.keras.layers import Dense, Input, Concatenate, Masking, Dropout, BatchNormalization
from tensorflow.keras import regularizers 
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pandas as pd
import numpy as np
import os
import time
from joblib import load
import matplotlib.pyplot as plt

"""all_labels = [
    "APPLE", "ORANGE", "BOOK", "COMPUTER", "PHONE", "CAR", "BUS",
    "HOUSE", "DOOR", "WINDOW", "TABLE", "CHAIR", "FAMILY", "FRIEND",
    "SCHOOL", "WORK", "TIME", "DAY", "MORNING", "PLEASE", "SORRY", "HELLO",
    "COME", "GO", "SIT", "READ", "WRITE", "LISTEN", "SEE", "COOK", "PLAY",
    "SELL", "OPEN", "CLOSE", "START", "STOP", "BIG", "SMALL", "LONG",
    "FAIL", "IMPROVE", "BETWEEN", "TOTAL", "GETTOGETHER", "WHY",
    "HAPPY", "SAD", "ANGRY", "LOVE", "BUY", "HOT", "COLD", "NEW", "OLD",
    "FAST", "SLOW", "UP", "DOWN", "IN", "OUT", "LIGHT", "GRAB", "YES", "NO",
    "EASY", "BOY", "GIRL", "COUGH", "MAYBE", "LAUGH"
]"""

# Create directory for models
os.makedirs("TunedModels", exist_ok=True)
os.makedirs("Results", exist_ok=True)

def wristNormalise(landmarks):
    wrist = landmarks[0]
    for i in range(len(landmarks)):
        landmarks[i][0] -= wrist[0]
        landmarks[i][1] -= wrist[1]
        landmarks[i][2] -= wrist[2]
    return landmarks



all_labels = ['LAST', 'TOMORROW', 'HORSE', 'THINK', 'IN', 'SLEEP', 'IMPORTANT', 'DIFFERENT', 'GOOD', 'SLOW', 'GRASS', 'FAST', 'FATHER', 'EMAIL', 'WEATHER', 'KITCHEN', 'SOME',
'INTERNET', 'GETTOGETHER', 'KNOW', 'NO', 'CAKE', 'OUT', 'WHY', 'TEXT', 'PLAY', 'WINTER', 'STREET', 'GAME', 'VEGETABLE', 'TRAIN', 'STOP', 'SPRING', 'EARTH', 'AFTER',
'SMALL', 'HAPPY', 'CAR', 'ANGRY', 'FIRE', 'NONE', 'ORANGE', 'SORRY', 'YESTERDAY', 'PARK', 'COME', 'COFFEE', 'LESS', 'MONTH', 'EASY', 'HOT', 'MOON', 'MOTHER', 'SIT',
'BASEBALL', 'NEXT', 'YEAR', 'WORK', 'METAL', 'TENNIS', 'BATHROOM', 'MORNING', 'LATER', 'TEA', 'FAMILY', 'RAIN', 'SELL', 'PICTURE', 'GO', 'CAMERA', 'BIRD', 'COUNTRY', 'DOOR', 'FOOTBALL',
'BOOK', 'NAME', 'SAD', 'TEACHER', 'HELLO', 'BEACH', 'WINDOW', 'LEARN', 'SUMMER', 'CLOSE', 'FRUIT',
'MORE', 'WHITE', 'APPLE', 'MANY', 'TRUE', 'UP', 'FRIEND', 'PLEASE', 'BROWN', 'FLOWER', 'GIRL', 'SILVER', 'SING', 'BIG', 'SISTER', 'HOME', 'CHAIR', 'SPORTS', 'COLOR',
'RESTAURANT', 'STAR', 'READ', 'UNDERSTAND', 'WATCH', 'MAYBE', 'DANCE', 'BUS', 'YELLOW', 'SEE', 'SCHOOL', 'STUDENT', 'RED', 'ANIMAL', 'PHONE', 'GREEN', 'SUN', 
'FIRST', 'WIND', 'COOK', 'BLUE', 'LOVE', 'YES', 'COLD', 'HELP', 'WATER', 'COOKIE', 'NEED', 'START', 'SHOES', 'BAD', 'OPEN', 'HOUSE', 'BOY', 'FISH', 'BEDROOM', 'NOW', 'TABLE', 'PEN', 'TREE', 'LONG', 'BEFORE',
'LISTEN', 'DAY', 'GOLD', 'NEW', 'WOOD', 'TIME', 'BROTHER', 'DOWN', 'ALL', 'BUY', 'OFFICE', 'WEEK', 'OLD', 'WRITE', 'FINISH', 'BIRTHDAY', 'COMPUTER', 'ICE']

# subset_labels will contain the first 70 labels
subset_labels = all_labels[:70]

# Load raw datasets that need filtering
train = pd.read_csv("trainss_164.csv")
val = pd.read_csv("valss_164.csv")
test = pd.read_csv("testss_164.csv")


train_df = train[train['Gloss'].isin(subset_labels)].reset_index(drop=True)
val_df = val[val['Gloss'].isin(subset_labels)].reset_index(drop=True)
test_df = test[test['Gloss'].isin(subset_labels)].reset_index(drop=True)


"""trainLabels = train["Gloss"].tolist()
valLabels = val["Gloss"].tolist()
testLabels = test["Gloss"].tolist()
trainLabels = list(set(trainLabels))
valLabels = list(set(valLabels))
testLabels = list(set(testLabels))

all_labels = list(set(trainLabels + valLabels + testLabels))"""


def format_data(data):
    num_frames = len(data)
    formatted = np.zeros((num_frames, 2, 21, 3))
    for i, frame in enumerate(data):
        for j in frame:
            side = 0 if j['side'] == 'Left' else 1
            landmarks = j['landmarks']
            formatted[i, side] = landmarks
    return formatted

def load_dataset(csv_file):
    left_hands = []
    right_hands = []
    labels_list = []
    
    #MPFrames_Joblib

    for idx, row in csv_file.iterrows():
        file = f"164LabelLandmarks/{row['Video file'].replace('.mp4', '.joblib')}"
        try:
            data = load(file)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
        data = format_data(data)
        
        left = data[:,0,:,:]  # left hand
        right = data[:,1,:,:]  # right hand
        
        left_hands.append(left)
        right_hands.append(right)
        labels_list.append(row["Gloss"])
        
    return left_hands, right_hands, labels_list

# Load all datasets
trainLeft, trainRight, trainLabels = load_dataset(train)
valLeft, valRight, valLabels = load_dataset(val)
testLeft, testRight, testLabels = load_dataset(test)

# Map labels to indices
nclasses = len(all_labels)

label_dict = {label: idx for idx, label in enumerate(all_labels)}

trainLabels = [label_dict[label] for label in trainLabels]
valLabels = [label_dict[label] for label in valLabels]
testLabels = [label_dict[label] for label in testLabels]


# Data processing functions
def Preprocess_Batch(batch_sequence):
    max_frames = max(len(sequence) for sequence in batch_sequence)
    batchSize = len(batch_sequence)
    padded = np.zeros((batchSize, max_frames, 63))

    for i, sequence in enumerate(batch_sequence):
        # Normalise wrist position
        sequence = wristNormalise(sequence)
        reshaped = sequence.reshape(len(sequence), -1)
        padded[i, :len(sequence)] = reshaped
    return padded

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

def evaluate_in_batches(model, leftHand, rightHand, labels, batch_size=32):
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    for start in range(0, len(leftHand), batch_size):
        end = min(start + batch_size, len(leftHand))
        
        lhBatch = [leftHand[i] for i in range(start, end)]
        rhBatch = [rightHand[i] for i in range(start, end)]
        labelBatch = [labels[i] for i in range(start, end)]
        
        leftPad = Preprocess_Batch(lhBatch)
        rightPad = Preprocess_Batch(rhBatch)
        labelBatch = to_categorical(labelBatch, num_classes=nclasses)
        
        metrics = model.test_on_batch(
            [leftPad, rightPad],
            labelBatch
        )
        total_loss += metrics[0]
        total_acc += metrics[1]
        num_batches += 1
    
    return total_loss/num_batches, total_acc/num_batches

def get_all_predictions(model, leftHand, rightHand):
    all_predictions = []
    
    for start in range(0, len(leftHand), 32):
        end = min(start + 32, len(leftHand))
        
        lhBatch = [leftHand[i] for i in range(start, end)]
        rhBatch = [rightHand[i] for i in range(start, end)]
        
        leftPad = Preprocess_Batch(lhBatch)
        rightPad = Preprocess_Batch(rhBatch)
        
        batch_preds = model.predict([leftPad, rightPad], verbose=0)
        all_predictions.extend(batch_preds)
    
    return np.array(all_predictions)

# Hyperparameter configurations to try
configs = [
    # Format: (dropout1, recurrent_dropout1, l2_1, dropout2, recurrent_dropout2, l2_2, dropout3, recurrent_dropout3, l2_3, dense_dropout, dense_l2)
    (0.1, 0.1, 0.005, 0.15, 0.1, 0.008, 0.2, 0.15, 0.01, 0.3, 0.001),    # Primary config
    (0.2, 0.1, 0.0005, 0.15, 0.1, 0.002, 0.2, 0.1, 0.005, 0.25, 0.001), # Lower regularization
    (0.1, 0.1, 0.005, 0.1, 0.15, 0.005, 0.2, 0.15, 0.005, 0.25, 0.003),     # Consistent L2
]   #(d,  r,   l2,  d,    r, l2,    d,    r, l2,   d,   l2)
# (_,_,l2,_,__,l2,_,_,l2,_,_) = configs[0]
config_prints = [
    "c0, Original config",
    "c1, Lower overall regularisation",
    "c2, Consistent L2 regularisation",
]

# Store results
results = []
best_accuracy = 0
best_model = None
best_config = None
best_model_path = None

# Prepare generators for training
trainGenerator = Batch_Passer(trainLeft, trainRight, trainLabels)
valGenerator = Batch_Passer(valLeft, valRight, valLabels)

epoch_steps = len(trainLeft) // 32 
val_steps = len(valLeft) // 32

# Loop through configurations
for config_idx, config in enumerate(configs):
    print(f"\n{'='*50}")
    print(f"Training configuration {config_idx+1}/{len(configs)}")
    print(f"{'='*50}")
    
    # Unpack configuration
    d1, rd1, l2_1, d2, rd2, l2_2, d3, rd3, l2_3, dense_d, dense_l2 = config
    
    # Create a unique name for this model
    timestamp = int(time.time())
    model_name = f"TunedModels/model_config{config_idx}_{timestamp}.h5"
    
    # Build model with this configuration
    leftIn = Input(shape=(None, 63), name="Left Hand")
    rightIn = Input(shape=(None, 63), name="Right Hand")
    
    leftMask = Masking(mask_value=0.)(leftIn)
    rightMask = Masking(mask_value=0.)(rightIn)
    
    # First LSTM layer with config parameters
    left1LSTM = LSTM(64, return_sequences=True, name="left_lstm1",
                    dropout=d1, recurrent_dropout=rd1,
                    kernel_regularizer=regularizers.l2(l2_1))(leftMask)
    right1LSTM = LSTM(64, return_sequences=True, name="right_lstm1",
                    dropout=d1, recurrent_dropout=rd1,
                    kernel_regularizer=regularizers.l2(l2_1))(rightMask)
    
    left1LSTM = BatchNormalization()(left1LSTM)
    right1LSTM = BatchNormalization()(right1LSTM)
    
    # Second LSTM layer
    left2LSTM = LSTM(64, return_sequences=True, name="left_lstm2",
                    dropout=d2, recurrent_dropout=rd2,
                    kernel_regularizer=regularizers.l2(l2_2))(left1LSTM)
    right2LSTM = LSTM(64, return_sequences=True, name="right_lstm2",
                     dropout=d2, recurrent_dropout=rd2,
                     kernel_regularizer=regularizers.l2(l2_2))(right1LSTM)
    
    left2LSTM = BatchNormalization()(left2LSTM)
    right2LSTM = BatchNormalization()(right2LSTM)
    
    # Third LSTM layer
    left3LSTM = LSTM(128, return_sequences=False, name="left_lstm3",
                    dropout=d3, recurrent_dropout=rd3,
                    kernel_regularizer=regularizers.l2(l2_3))(left2LSTM)
    right3LSTM = LSTM(128, return_sequences=False, name="right_lstm3",
                     dropout=d3, recurrent_dropout=rd3,
                     kernel_regularizer=regularizers.l2(l2_3))(right2LSTM)
    
    combine = Concatenate(name="combined")([left3LSTM, right3LSTM])
    X = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(dense_l2))(combine)
    X = Dropout(dense_d)(X)
    output = Dense(nclasses, activation='softmax')(X)
    
    architecture = Model(inputs=[leftIn, rightIn], outputs=output)
    architecture.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    architecture.summary()
    
    # Define callbacks
    checkpoint = ModelCheckpoint(model_name, monitor='val_accuracy',
                               verbose=1, save_best_only=True, mode='max')
    
    slowLR = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.9,
        patience=10,
        verbose=1,
        min_lr=1e-6
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=60,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    print(f"Training model with configuration:")
    print(f"  LSTM1: dropout={d1}, recurrent_dropout={rd1}, l2={l2_1}")
    print(f"  LSTM2: dropout={d2}, recurrent_dropout={rd2}, l2={l2_2}")
    print(f"  LSTM3: dropout={d3}, recurrent_dropout={rd3}, l2={l2_3}")
    print(f"  Dense: dropout={dense_d}, l2={dense_l2}")
    
    history = architecture.fit(
        trainGenerator, validation_data=valGenerator,
        steps_per_epoch=epoch_steps, epochs=260,
        validation_steps=val_steps,
        callbacks=[checkpoint, slowLR, early_stop])
    
    # Load the best weights (saved by checkpoint callback)
    architecture.load_weights(model_name)
    
    # Evaluate on test set
    test_loss, test_acc = evaluate_in_batches(architecture, testLeft, testRight, testLabels)
    print(f"\nConfiguration {config_idx+1} Test Accuracy: {test_acc*100:.2f}%")
    
    # Record results
    val_acc = max(history.history['val_accuracy'])
    results.append({
        'config_idx': config_idx,
        'config': config,
        'test_acc': test_acc,
        'val_acc': val_acc,
        'model_path': model_name
    })
    
    # Check if this is the best model so far
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        best_model = architecture
        best_config = config
        best_model_path = model_name
        print(f"New best model found! Test accuracy: {best_accuracy*100:.2f}%")

label_map = {label: i for i, label in enumerate(all_labels)}



# Print summary of all results
print("\n\n" + "="*60)
print("HYPERPARAMETER TUNING RESULTS")
print("="*60)
results.sort(key=lambda x: x['test_acc'], reverse=True)
for i, result in enumerate(results):
    print(f"Rank {i+1}: Config {result['config_idx']} - Test: {result['test_acc']*100:.2f}%, Val: {result['val_acc']*100:.2f}%")

# Save best model to the final location
if best_model is not None:
    print(f"\nBest configuration: {best_config}")
    print(f"Best configs index: {results[0]['config_idx']}")
    print(f"Best model path: {best_model_path}")
    print(f"Best test accuracy: {best_accuracy*100:.2f}%")
    print(config_prints)
    # Save the best model
    os.makedirs("ModelsAsc", exist_ok=True)
    best_model.save("ModelsAsc/70_labels64to256_tuned.h5", overwrite=True)
    print("Best model saved to Models/Version2LSTM_tuned.h5")
    
    # Generate predictions for confusion matrix and classification report
    print("\n--- Generating Classification Report for Best Model ---")
    print("Generating predictions...")
    y_pred_prob = get_all_predictions(best_model, testLeft, testRight)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = testLabels
    
    # Create classification report
    report = classification_report(
        y_true, 
        y_pred,
        labels=np.arange(len(all_labels)),
        target_names=all_labels,
        digits=3
    )
    
    # Print and save the report
    print("\nClassification Report:")
    print(report)
    
    with open(f"Results/classification_report_best_config{results[0]['70LabelAscArch_idx']}.txt", "w") as f:
        f.write("Classification Report\n")
        f.write(f"Model: Best configuration {results[0]['_70LabelAscArch_idx']}\n")
        f.write(f"Test Accuracy: {best_accuracy*100:.2f}%\n\n")
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
    
    # Generate confusion matrix visualization
    print("\nGenerating confusion matrix visualization...")
    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(np.arange(len(all_labels)), all_labels, rotation=90, fontsize=8)
    plt.yticks(np.arange(len(all_labels)), all_labels, fontsize=8)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'Results/confusion_matrix_best_config{results[0]["70LabelAscArch_idx"]}.png', dpi=150)
    plt.close()
else:
    print("No models were successfully trained.")