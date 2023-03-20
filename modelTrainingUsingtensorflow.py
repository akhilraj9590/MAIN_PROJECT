#!/usr/bin/env python
# coding: utf-8

# In[13]:


import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


file='E:\MCA-2 Year\Main Project\dataset\Mayamalavagowla\Mmg.wav'
audio, sample_rate = librosa.load(file)
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
print("mfccs",mfccs_features)
print("mfccs scaled",mfccs_scaled_features)


def features_extractor(file):
    audio, sample_rate = librosa.load(file)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features


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
# extracted_features_df


# In[14]:


### Split the dataset into independent and dependent dataset
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())


# In[15]:


X.shape


# In[16]:


X


# In[18]:


### Label Encoding
###y=np.array(pd.get_dummies(y))
### Label Encoder
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))


# In[19]:


y


# In[20]:


y.shape


# In[21]:


### Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[22]:


print("ddz",y_train)
X_test


# In[23]:


X_train.shape


# In[24]:


X_test.shape


# In[25]:


y_train.shape


# In[26]:


y_test.shape


# In[27]:


import tensorflow as tf
print(tf.__version__)


# In[28]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics


# In[29]:


### No of classes
num_labels=y.shape[1]


# In[30]:


num_labels


# In[31]:


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


# In[32]:


model.summary()


# In[33]:


model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')


# In[34]:


## Trianing my model
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 

num_epochs = 100
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)


# In[35]:


test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])


# In[36]:


X_test[1]


# In[37]:


predictions = (model.predict(X_test) > 0.5).astype("int32")
predictions


# In[46]:


# filename="E:\MCA-2 Year\Main Project\dataset\Mayamalavagowla\Mmg.wav"
# audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
# mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
# mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

# mfccs_scaled_features.shape
filename="E:\MCA-2 Year\Main Project\dataset\Mayamalavagowla\Mmg.wav"
# filename="E:\MCA-2 Year\Main Project\dataset\Mayamalavagowla\mayaShorts.wav"
audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

print(mfccs_scaled_features)
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
print(mfccs_scaled_features)
print(mfccs_scaled_features.shape)
predict_x=model.predict(mfccs_scaled_features) 
classes_x=np.argmax(predict_x,axis=1)
print(classes_x)
prediction_class = labelencoder.inverse_transform(classes_x) 
prediction_class


# In[ ]:


mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
print(mfccs_scaled_features)


# In[ ]:


mfccs_scaled_features.shape
mfccs_scaled_features
# X_train


# In[ ]:


# predicted_label=model.predict_class(mfccs_scaled_features)
# print(predicted_label)
y_pred = model.predict(X_train)
X_test.shape
# 'y_pred' will be an array of predicted probabilities for each class in your output
# to get the predicted class for each input, you can use the 'argmax' function to find the index of the maximum probability in each row
# y_pred_classes = y_pred.argmax(axis=-1)
# y_pred_classes
a=labelencoder.inverse_transform(y_pred_classes)
a


# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# X_test
# calculate evaluation metrics
# accuracy = accuracy_score(y_test, y_train)
# accuracy
# precision = precision_score(y_test, y_train)
# recall = recall_score(y_test, y_train)
# f1 = f1_score(y_test, y_train)

# print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1 score: {f1:.2f}")


# In[ ]:





# In[ ]:




