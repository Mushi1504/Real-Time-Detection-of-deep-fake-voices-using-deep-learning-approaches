#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries
import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
#from tqdm import tqdm  # Add tqdm for progress bar
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score


# In[2]:


import joblib
from sklearn.metrics import classification_report

# Save the trained Random Forest model to a file
model_filename = "random_forest_model.pkl"
#joblib.dump(clf, model_filename)
#print(f"Model saved as {model_filename}")

# Load the model from the saved file
loaded_model = joblib.load(model_filename)
print("Model loaded successfully!")


# In[3]:


# Function to extract features from an audio file
def extract_features(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract MFCC (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    # Extract additional features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    rmse = np.mean(librosa.feature.rms(y=y))

    # Combine all features into a single vector
    features = np.hstack([mfccs_mean, spectral_centroid, chroma, zero_crossing_rate, rmse])
    
    return features


# In[ ]:




# Generate a classification report on the test set
#y_pred = loaded_model.predict(X_test)
#report = classification_report(y_test, y_pred, target_names=["Real", "Fake"])
#print("Classification Report:\n", report)

# Function to predict fake or real for a new .wav file
def predict_fake_or_real(wav_file, model):
    # Extract features from the provided .wav file
    features = extract_features(wav_file)
    features = features.reshape(1, -1)  # Reshape for prediction
    prediction = model.predict(features)

    if prediction[0] == 1:
        return "Fake"
    else:
        return "Real"
def voice_detector(input_voice):
                    # Ask the user to provide a .wav file and predict its class
                    user_wav_file = input_voice

                    # Check if the file exists
                    if os.path.exists(user_wav_file) and user_wav_file.endswith(".wav"):
                        result = predict_fake_or_real(user_wav_file, loaded_model)
                        print(f"The provided file is classified as: {result}")
                    else:
                        print("The provided file path is either invalid or not a .wav file.")


                    return   result



