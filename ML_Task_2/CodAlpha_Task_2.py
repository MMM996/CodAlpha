# This is the code for Task 2 Cod Alpha Internship
# Task 2: Speech Emotion Recognition
# The Code is written by Muhammad Mudassir Majeed
# The date is Jan-24.
# Dataset: https://www.kaggle.com/datasets/parisrohan/credit-score-classification
# Good Resource: https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition
# Resource 2: https://paperswithcode.com/task/speech-emotion-recognition

#-----------------------------------------------------------------------------#

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Part 1: Load Dataset
import os
import sys

# We will be using three audio data sets. These are amended verisions of original sets

# Ravdess
Ravdess_path = 'Task 2\\data\Ravdess\\audio_speech_actors_01-24\\'
ravdess_dir_list = os.listdir(Ravdess_path)

file_emotion = []
file_path = []

for direc in ravdess_dir_list:
    actor = os.listdir(Ravdess_path + direc)
    for file in actor:
        part = file.split('-')[2]
        file_emotion.append(part)
        file_path.append(Ravdess_path + direc + '\\' + file)
        
emotions_df = pd.DataFrame(file_emotion, columns = ['Emotions'])

paths_df = pd.DataFrame(file_path, columns=['Path'])
Ravdess_df = pd.concat([emotions_df, paths_df], axis=1)
Ravdess_df.info()

Ravdess_df['Emotions'] = pd.to_numeric(Ravdess_df['Emotions'])

# Actual Emotions
Ravdess_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
Ravdess_df.head()


# Crema 
Crema_path = 'Task 2\\data\\Crema\\'

Crema_dir_list = os.listdir(Crema_path)
file_emotion = []
file_path = []

for file in Crema_dir_list:
    part = file.split('_')[2]
    if part == 'SAD':
        file_emotion.append('sad')
    elif part == 'ANG':
        file_emotion.append('angry')
    elif part == 'DIS':
        file_emotion.append('disgust')
    elif part == 'FEA':
        file_emotion.append('fear')
    elif part == 'HAP':
        file_emotion.append('happy')
    elif part == 'NEU':
        file_emotion.append('neutral')
    else:
        file_emotion.append('unknown')
        
    file_path.append(Crema_path + file)
    
emotions_df = pd.DataFrame(file_emotion, columns=['Emotions'])
paths_df = pd.DataFrame(file_path, columns=['Path'])
Crema_df = pd.concat([emotions_df, paths_df], axis = 1)


# Tess
Tess_path = 'Task 2\\data\\Tess\\'

Tess_dir_list = os.listdir(Tess_path)
file_emotion = []
file_path = []

for direc in Tess_dir_list:
    directory_list = os.listdir(Tess_path + direc)
    for file in directory_list:
        part = file.split('.')[0]
        part = part.split('_')[2]
        file_path.append(Tess_path + direc + '\\' + file)
        file_emotion.append(part)
        

emotions_df =pd.DataFrame(file_emotion, columns=['Emotions'])
paths_df = pd.DataFrame(file_path, columns= ['Path'])
Tess_df = pd.concat([emotions_df, paths_df], axis = 1)

Tess_df.Emotions.replace({'ps':'surprise'}, inplace=True)
Tess_df.head()

data_file = pd.concat([Ravdess_df, Crema_df, Tess_df], axis = 0)
data_file.to_csv('Task 2\\data.csv', index = False)

data = pd.read_csv('Task 2\\data.csv')

# Part 2: Preliminary EDA
data.info()
data['Emotions'].value_counts()

# Observe Class Balance
sns.countplot(x=data['Emotions'], label = 'Count')


# Spectogram for Audio
import librosa
import librosa.display
from IPython.display import Audio


emotion = 'happy'
path = np.array(data.Path[data.Emotions==emotion])[1]
data_emotion, sampling_rate = librosa.load(path)
X = librosa.stft(data_emotion)
Xdb = librosa.amplitude_to_db(abs(X))
plt.title('Spectogram for audio with {} emotion'.format(emotion))
librosa.display.specshow(Xdb, sr=sampling_rate, x_axis='time', y_axis='hz')
plt.colorbar()

Audio(path)

# Part 3: Pre-processing
# Data Augmentation
# We can perform noise addition, Streching, Changing pitch, etc.
# No Augmentation performed for this task

# Part 4: Feature Extraction
# Define X and Y

X, Y = [], []


for path, emotion in zip(data.Path, data.Emotions):
   data_current, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
   result = np.array([])
   
   # ZCR
   zcr = np.mean(librosa.feature.zero_crossing_rate(y=data_current).T, axis=0)
   result=np.hstack((result, zcr))
   
   # Chroma_stft
   stft = np.abs(librosa.stft(data_current))
   chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
   result = np.hstack((result, chroma_stft))
   
   # MFCC
   mfcc = np.mean(librosa.feature.mfcc(y=data_current, sr=sample_rate).T, axis=0)
   result = np.hstack((result, mfcc)) 
   
   # RMS Values
   rms = np.mean(librosa.feature.rms(y=data_current).T, axis=0)
   result = np.hstack((result, rms))
   
   # MelSpectogram
   mel = np.mean(librosa.feature.melspectrogram(y=data_current, sr=sample_rate).T, axis=0)
   result = np.hstack((result, mel))
   
   # Entropy of Energy
   energy_entropy = np.mean(librosa.feature.spectral_flatness(S=stft).T, axis=0)
   result = np.hstack((result, energy_entropy))
    
   # Spectral Centroid
   centroid = np.mean(librosa.feature.spectral_centroid(y=data_current, sr=sample_rate).T, axis=0)
   result = np.hstack((result, centroid))
    
   # Spectral Rolloff
   rolloff = np.mean(librosa.feature.spectral_rolloff(y=data_current, sr=sample_rate).T, axis=0)
   result = np.hstack((result, rolloff))
    
   result_final = np.array(result)
   X.append(result_final)
   Y.append(emotion)
   
Features = pd.DataFrame(X)
Features['labels'] = Y
Features.to_csv('Task 2\\features.csv', index=False)

Features.head()

# Split into X and Y
data_features = pd.read_csv('Task 2\\features.csv')

X_feat = data_features.iloc[: ,:-1].values
Y_feat = data_features['labels'].values

# Normalize Data
from sklearn.preprocessing import MinMaxScaler
norm = MinMaxScaler()
X_final = norm.fit_transform(X_feat)

# Convert Output Emotions to Numerical Value
# We can use label encoding or one hot encoding
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
Y_final = label.fit_transform(Y_feat)

# Part 5: Model Building
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y_final, test_size=0.2, 
                                                    stratify= Y_final, random_state=42)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50)
rf.fit(X_train, Y_train)

rf_predict = rf.predict(X_test)

# Decision Trees
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(X_train, Y_train)

DT_predict = rf.predict(X_test)


# DL Model
import keras
from keras.models import Sequential
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical

X_train_DL = np.expand_dims(X_train, axis=2)
X_test_DL = np.expand_dims(X_test, axis=2)
Y_train_DL = Y_train
Y_test_DL = Y_test

X_train_DL.shape, X_test_DL.shape, Y_train_DL.shape, Y_test.shape

# Model 

model=Sequential()
model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(X_train_DL.shape[1], 1)))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
model.add(Dropout(0.3))

model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

model.add(Conv1D(32, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=8, activation='softmax'))

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x= X_train_DL, y= Y_train_DL, batch_size = 10, epochs = 50,
          shuffle = True, verbose = 2, validation_split = 0.1)

predictions = model.predict(x= X_test_DL, batch_size = 10, verbose = 0)
    
rounded_predictions = np.argmax(predictions, axis =1)
DL_predict = rounded_predictions


# Part 6: Model Evaluation
from sklearn.metrics import confusion_matrix, classification_report

# Confusion Matrix
confusion_matrix_DL = confusion_matrix(Y_test_DL, DL_predict)
confusion_matrix_rf = confusion_matrix(Y_test, rf_predict)
confusion_matrix_DT = confusion_matrix(Y_test, DT_predict)

# Classification Report
classification_report_DL = classification_report(Y_test_DL, DL_predict)
classification_report_rf = classification_report(Y_test, rf_predict)
classification_report_DT = classification_report(Y_test, DT_predict)

print("Deep Learning Confusion Matrix:\n", confusion_matrix_DL)
print("Deep Learning Classification Report:\n", classification_report_DL)

print("Random Forest Confusion Matrix:\n", confusion_matrix_rf)
print("Random Forest Classification Report:\n", classification_report_rf)

print("Decision Tree Confusion Matrix:\n", confusion_matrix_DT)
print("Decision Tree Classification Report:\n", classification_report_DT)


