# training CNN with ESC-50 dataset
# from https://qiita.com/cvusk/items/61cdbce80785eaf28349
import os
#import random
import numpy as np
import pandas as pd
import librosa
import librosa.display
#import matplotlib.pyplot as plt
#import seaborn as sn
from sklearn import model_selection
#from sklearn import preprocessing
import math

# define directories
base_dir = "F:\dataset/ESC-50-master"
# esc_dir = os.path.join(base_dir, "ESC-50")
meta_path = os.path.join(base_dir, "meta/esc50.csv")
audio_dir = os.path.join(base_dir, "audio/")

# load metadata
meta_data = pd.read_csv(meta_path)
data_size = meta_data.shape
print(data_size)
num_data = data_size[0]

# combine label No. and label name 
class_dict = {}
for i in range(num_data):
    if meta_data.loc[i, "target"] not in class_dict.keys():
        class_dict[meta_data.loc[i,"target"]] = \
        meta_data.loc[i,"category"]

class_pd = pd.DataFrame(
        list(class_dict.items()), 
        columns=["labels","classes"])    
 

# LOAD DATASET
# load from filename
def load_wave_data(audio_dir, file_name):
    file_path = os.path.join(audio_dir, file_name)
    x, fs = librosa.load(file_path, sr=44100)
    return x,fs


#min-max normalization
def normalize(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

# convert wave data to mel-stft
def calc_melsp(x, n_fft=1024, hop_length=256):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
    return melsp

# AUGUMENTATION
#
def add_white_noise(x, rate=0.002):
    return x + rate*np.random.randn(len(x))

def shift_sound(x, rate=2):
    return np.roll(x, int(len(x)//rate))

def stretch_sound(x, rate=1.1):
    input_length = len(x)
    x = librosa.effects.time_stretch(x, rate)
    if len(x)>input_length:
        return x[:input_length]
    else:
        return np.pad(x, (0, max(0, input_length - len(x))), "constant")


# SPRIT DATASET
#
# get training dataset and target dataset
x = list(meta_data.loc[:,"filename"]) #filename
y = list(meta_data.loc[:, "target"])  #label No.

x_train, x_test, y_train, y_test = \
    model_selection.train_test_split(x, y, test_size=0.20, stratify=y)
print("x train:{0}\ny train:{1}\nx test:{2}\ny test:{3}"
    .format(len(x_train), len(y_train), len(x_test), len(y_test)))


# SAVE DATASET
#
# convert wave list to melsp array
freq = 128  # melsp_dim(mels) = 128
fs = 44100
max_leng = 5 #sec
hop_length = 256
time = math.ceil(max_leng*fs/hop_length) # = 862

# save wave data in npz, with augmentation
def save_np_data(filename, x, y, aug=None, rates=None):
    np_data = np.zeros(freq*time*len(x)).reshape(len(x), freq, time)
    np_targets = np.zeros(len(y))
    for i in range(len(y)):
        _x, fs = load_wave_data(audio_dir, x[i])
        if aug is not None:
            _x = aug(x=_x, rate=rates[i])
        _x = calc_melsp(_x)
        np_data[i] = _x
        np_targets[i] = y[i]
        
    np.savez(filename, x=np_data, y=np_targets)
    
# test(eval) data
if not os.path.exists("esc_melsp_test.npz"):
    save_np_data("esc_melsp_test.npz", x_test,  y_test)
    print("exported test data")

# raw train data
if not os.path.exists("esc_melsp_train_raw.npz"):
    save_np_data("esc_melsp_train_raw.npz", x_train,  y_train)
    print("exported train data")
    
# train data: white noise
if not os.path.exists("esc_melsp_train_wn.npz"):
    rates = np.random.randint(1,50,len(x_train))/10000
    save_np_data("esc_melsp_train_wn.npz", 
                 x_train,  y_train, aug=add_white_noise, rates=rates)
    print("exported train data: white noise")

# train data: sound shift
if not os.path.exists("esc_melsp_train_ss.npz"):
    rates = np.random.choice(np.arange(2,6),len(y_train))
    save_np_data("esc_melsp_train_ss.npz", 
                 x_train,  y_train, aug=shift_sound, rates=rates)
    print("exported train data: sound shift")

# train data: stretch
if not os.path.exists("esc_melsp_train_st.npz"):
    rates = np.random.choice(np.arange(80,120),len(y_train))/100
    save_np_data("esc_melsp_train_st.npz", 
                 x_train,  y_train, aug=stretch_sound, rates=rates)
    print("exported train data: stretch")

# train data: white noise AND shift OR stretch (combination)
if not os.path.exists("esc_melsp_train_com.npz"):
    np_data = np.zeros(freq*time*len(x_train)).reshape(len(x_train), freq, time)
    np_targets = np.zeros(len(y_train))
    for i in range(len(y_train)):
        x, fs = load_wave_data(audio_dir, x_train[i])
        x = add_white_noise(x=x, rate=np.random.randint(1,50)/1000)       
        if np.random.choice((True,False)):
            x = shift_sound(x=x, rate=np.random.choice(np.arange(2,6)))
        else:
            x = stretch_sound(x=x, rate=np.random.choice(np.arange(80,120))/100)
        
        x = calc_melsp(x)
        np_data[i] = x
        np_targets[i] = y_train[i]
    
    np.savez("esc_melsp_train_com.npz", x=np_data, y=np_targets)
    print("exported train data: combination")
    
print("preprocess finished!")
