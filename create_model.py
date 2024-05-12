# import librosa
# import soundfile
import os, glob, pickle
# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from utils import load_data

emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}


emotions_to_observe = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Surprised']


x,y=load_data("C:/Users/HP/Desktop/speech_emotion_detection/speech_data/Actor_*/*.wav",emotions_to_observe)


model=MLPClassifier(batch_size=256, max_iter=500)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2 , random_state=24)


model.fit(x_train,y_train)

y_pred=model.predict(x_test)


accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print(f"Accuracy of Model {accuracy*100}")
print(f"Accuracy of Random Guessing {1/len(emotions_to_observe)*100}")

pickle.dump(model, open("model.model", "wb"))
