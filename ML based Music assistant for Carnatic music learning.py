import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np



def mfccs(file_path):
    # Load audio file using librosa
    audio, sample_rate = librosa.load(file_path)
    # Extract mfccs using librosa
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
    return mfccs


def spectral_contrast(file_path):
    # Load audio file using librosa
    audio, sample_rate = librosa.load(file_path)
    # Extract spectral_contrast using librosa
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    return spectral_contrast


def spectral_centroid(file_path):
    # Load audio file using librosa
    audio, sample_rate = librosa.load(file_path)
    # Extract spectral_centroid using librosa
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    return spectral_centroid
    
matrix=np.array([])
audio_path='E:\MCA-2 Year\Main Project\dataset'
for dirpath, dirnames, filenames in os.walk(audio_path):
        for filename in filenames:
            if filename.endswith('.wav'):
                filepath=os.path.join(dirpath, filename)
                x=spectral_centroid(filepath).mean()
                y=spectral_contrast(filepath).mean()
                z=mfccs(filepath).mean()
                label=filepath.replace("E:\MCA-2 Year\Main Project\dataset", '').replace(filename,"").replace('\\','')
                new_row = np.array([x,y,z,label])
                new_array = np.append(matrix, [new_row])
                matrix = new_array
                
            
print(matrix)

# Creating a DataFrame
DataSample= matrix
SimpleDataFrame=pd.DataFrame(data=DataSample, columns=['spectral_contrast','spectral_centroid','mfccs',label])
print(SimpleDataFrame)
 
# # Exporting data frame to a csv/Excel file
# # Many other options are available which can be seen using dot tab option
 
# # Exporting data as a csv file
# SimpleDataFrame.to_csv('E:\MCA-2 Year\Main Project\SimpleDataFrame.csv')