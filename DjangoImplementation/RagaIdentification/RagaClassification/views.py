from django.shortcuts import render
from django.conf import settings
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 
import keras
from django.views.decorators.csrf import csrf_exempt
# import pydub
import io
from django.contrib import messages
from django.http import HttpResponse
# from django.core.files.storage import FileSystemStorage
from pydub import AudioSegment
import pyaudio
import wave









def features_extractor(file):
    audio, sample_rate = librosa.load(file)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features

def makeDataset():
    audio_path='audio_dataset'
    extracted_features=[]
    for dirpath, dirnames, filenames in os.walk(audio_path):
            for filename in filenames:
                if filename.endswith('.wav'):
                    filepath=os.path.join(dirpath, filename)
                    label=filepath.replace("audio_dataset", '').replace(filename,"").replace('\\','')
                    mfccs=features_extractor(filepath)
                    extracted_features.append([mfccs,label])
                    

    extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
    print('hai',extracted_features_df)
    extracted_features_df.to_pickle('RagamClassificationDataset1.pkl')



extracted_features_df=pd.read_pickle('RagamClassificationDataset.pkl')
### Split the dataset into independent and dependent dataset
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())
### Label Encoding
###y=np.array(pd.get_dummies(y))
### Label Encoder

labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))
### Train Test Split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


### No of classes
num_labels=y.shape[1]


def makeModel():
    model=Sequential()
    ###first layer
    model.add(Dense(100,input_shape=(40,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    ###second layer
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    ###third layer
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    ###final layer
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
    ## Trianing my model


    num_epochs = 100
    num_batch_size = 32

    checkpointer = ModelCheckpoint(filepath='saved_models/Model_classification.hdf5', 
                                verbose=1, save_best_only=True)
    start = datetime.now()

    model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)


    duration = datetime.now() - start
    print("Training completed in time: ", duration)


# Load the model from file
model = keras.models.load_model('saved_models/audio_classification85.hdf5')


# filename="C:/Users/Akhilraj/Downloads/Natta10M.wav"
# audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
# mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
# mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

# # print(mfccs_scaled_features)
# mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
# # print(mfccs_scaled_features)
# # print(mfccs_scaled_features.shape)
# predict_x=model.predict(mfccs_scaled_features) 
# classes_x=np.argmax(predict_x,axis=1)
# print(classes_x)
# prediction_class = labelencoder.inverse_transform(classes_x) 
# print(prediction_class)
# # test_loss, test_accuracy = model.evaluate(X_test, y_test)
# # print(test_accuracy)

def home(request):
    return render(request,'home.html')

def uploadAudio(request):
     if request.method == 'POST':
        uploaded_file = request.FILES['audio']
        file_name = uploaded_file.name.lower()
        if not (file_name.endswith('.wav') or file_name.endswith('.mp3') or file_name.endswith('.flac') or file_name.endswith('.ogg')):
            return render(request, 'error.html', {'error': 'Invalid file format. Only WAV, MP3, FLAC, and OGG are supported.'})
        audio, sample_rate = librosa.load(uploaded_file) #, res_type='kaiser_fast'
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
        mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
        predict_x=model.predict(mfccs_scaled_features)
        classes_x=np.argmax(predict_x,axis=1)
        results = labelencoder.inverse_transform(classes_x)
        raga="Predicted raga is :"

        return render(request, 'uploadAudio.html', {'results': results,'raga':raga})
     return render(request,'uploadAudio.html')



@csrf_exempt
def predictByRecord(request):      
    if request.method == 'POST':
        audio_file = request.FILES.get('audio_file')
        if audio_file:
            # Save the audio file to a temporary location
            with open('tmp/audio.wav', 'wb+') as destination:
                for chunk in audio_file.chunks():
                    destination.write(chunk)
           
            # sound = AudioSegment.from_mp3(input_file)
            # audio, sample_rate = librosa.load(uploaded_file,sr=22050, res_type='kaiser_fast')
            # print("asjjh",audio)
        # d="predicted raga is Mayamalavagowla"
        # return render(request,'recordAudio.html',{"d":d})
            
    return render(request,'recordAudio.html')



def record_audio(request):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 10
    WAVE_OUTPUT_FILENAME = "audio.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    audio, sample_rate = librosa.load('E:/MCA-2 Year/Main Project/code/MAIN_PROJECT/DjangoImplementation/RagaIdentification/audio.wav') #, res_type='kaiser_fast'
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    predict_x=model.predict(mfccs_scaled_features)
    if np.max(predict_x) > 0.8:
        classes_x=np.argmax(predict_x,axis=1)
        results = labelencoder.inverse_transform(classes_x) 
        raga="Predicted raga is :"
        return render(request, 'recordAudio.html', {'results': results,'raga':raga})
    else:
        raga='No raga identified'
        results='.'
        return render(request, 'recordAudio.html', {'results': results,'raga':raga})
    
    
    

    # Further process the audio here
    






loss, accuracy = model.evaluate(X_test, y_test)

# Print the evaluation metrics
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")
from sklearn.metrics import confusion_matrix

# Convert y_test to label-encoded format
y_test_labels = np.argmax(y_test, axis=1)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Convert y_pred to label-encoded format
y_pred_labels = np.argmax(y_pred, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(y_test_labels, y_pred_labels)
print("confusion_matrix",cm)

# from sklearn.metrics import roc_curve, auc

# # Convert y_test to label-encoded format
# y_test_labels = np.argmax(y_test, axis=1)

# # Make predictions on the test data
# y_pred = model.predict(X_test)

# # Compute ROC curve and AUC for each class
# n_classes = y_test.shape[1]
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test_labels[:, i], y_pred[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
    
# # Compute micro-average ROC curve and AUC
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test_labels.ravel(), y_pred.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# import matplotlib.pyplot as plt

# # Print ROC curves and AUC for each class
# for i in range(n_classes):
#     plt.figure()
#     plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic for Class {}'.format(i))
#     plt.legend(loc="lower right")
#     plt.show()

# # Print micro-average ROC curve and AUC
# plt.figure()
# plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = %0.2f)' % roc_auc["micro"])
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (micro-average)')
# plt.legend(loc="lower right")
# plt.show()
