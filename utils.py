import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


emotions={
  '01':'Neutral',
  '02':'Calm',
  '03':'Happy',
  '04':'Sad',
  '05':'Angry',
  '06':'Fearful',
  '07':'Disgusted',
  '08':'Surprised'
}

def load_data(sound_directory, emotions_to_observe):

    
    x,y=[],[]
    for file in glob.glob(sound_directory):
        file_name=os.path.basename(file)
        print(file_name)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in emotions_to_observe:
            continue
        feature=extract_feature(file)
        x.append(feature) #Feature Vector
        y.append(emotion) #Class Variable

    return np.array(x),y

def extract_feature(file_name):

    
    if not os.path.exists(file_name):
        return "File Doesn't Exist"

    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate

        #Processing signal in the time-frequency domain
        try:
            stft=np.abs(librosa.stft(X))
        except Exception as e:
            print(e)
            return

        #Initializing result array of lists
        result=np.array([])
        try:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        except Exception as e:
            print(e)
            return
        try:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        except Exception as e:
            print(e)
            return
        try:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
        except Exception as e:
            print(e)
            return
        return result

def convert(audio_path):
    
    if not os.path.exists(audio_path):
	       return "File Doesn't Exist"
    file_split_list = audio_path.split("/")
    filename = file_split_list[-1].split(".")[0]
    new_filename = f"{filename}_converted.wav"
    file_split_list[-1] = new_filename
    seperator = "/"
    target_path = seperator.join(file_split_list)
    if not audio_path.endswith(".wav"):
        return "Invalid File: Must be in .wav format"
    else:
        try:
            os.system(f"ffmpeg -i {audio_path} -ac 1 -ar 16000 {target_path}")
        except Exception as e:
            print(e)
            return
        return target_path
