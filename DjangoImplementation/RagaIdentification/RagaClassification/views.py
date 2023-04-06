from django.shortcuts import render
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





def features_extractor(file):
    audio, sample_rate = librosa.load(file)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features

def makeDataset():
    audio_path='E:\MCA-2 Year\Main Project\dataset'
    extracted_features=[]
    for dirpath, dirnames, filenames in os.walk(audio_path):
            for filename in filenames:
                if filename.endswith('.wav'):
                    filepath=os.path.join(dirpath, filename)
                    label=filepath.replace("E:\MCA-2 Year\Main Project\dataset", '').replace(filename,"").replace('\\','')
                    mfccs=features_extractor(filepath)
                    extracted_features.append([mfccs,label])
                    

    extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
    print('hai',extracted_features_df)
    extracted_features_df.to_pickle('RagamClassificationDataset.pkl')


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
model = keras.models.load_model('saved_models/Model_classification.hdf5')


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
     if request.method == 'POST':
        uploaded_file = request.FILES['audio']
        audio, sample_rate = librosa.load(uploaded_file, res_type='kaiser_fast')
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
        mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
        predict_x=model.predict(mfccs_scaled_features)
        classes_x=np.argmax(predict_x,axis=1)
        prediction_class = labelencoder.inverse_transform(classes_x)
        print(prediction_class)
     return render(request,'home.html')