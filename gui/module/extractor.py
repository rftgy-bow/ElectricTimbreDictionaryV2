# coding: utf-8
#特徴量抽出
print("***Importing tensorflow modules...***")
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import math
from sklearn.decomposition import PCA

#svm_path = "./saved_model2"

fs = 44100
hop_length = 256
max_leng = 5
time = math.ceil(max_leng*fs/hop_length)

# 特徴量ベクトルに対応するファイル名はどこで保持する？

# convert wave data to fft
def _calc_melsp(x, n_fft=1024, hop_length=256):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
    return melsp

# MAIN
# extract features from  audio files
# audio_paths(N) -> features_list (128xN)
def prepareModel(svm_path):
    print("***Loading saved_model...***")
    temp_model = load_model(svm_path)
    return temp_model
    
def _trimming(wave, fs, max_leng):
    powers = pow(wave, 2)
    max_index = np.argmax(powers)
    win_width = fs*max_leng
    if(len(wave) - max_index >= win_width): 
        #max_indexより後ろに十分な長さがあるとき
        wave = wave[max_index : max_index+win_width]
    else:
        #十分な長さがないとき -> 後ろから5秒分取り出す
        wave = wave[-1*win_width:]
    
    return wave

def extractFeatures(audio_paths, temp_model):
    features_list = []
    for audio_path in audio_paths:
        #fs = 44.1kHzに自動で変換
        wave, sr = librosa.load(audio_path, fs)
        
        if(len(wave) > fs*max_leng):
            #5秒以上のものは音量最大のフレームを基準にトリミング
            print("###Detected: wav file longer than 5sec###")
            wave = _trimming(wave, fs, max_leng)
        
        # メルスペクトログラムに変換
        spec = _calc_melsp(wave)
        
        ## 5秒以下のものはpadding
        pad_length = time - spec.shape[1]
        pad_spec = np.pad(spec,[(0,0),(0,pad_length)],'constant')
        
        # ndarrayに次元を追加する
        pad_spec = pad_spec[None, ..., None]
        
        # 特徴ベクトルの抽出
        prediction = temp_model.predict(pad_spec)
        features = prediction[1]
        
        features_list.append(features)
        print("***Extracted Features: ", audio_path, "***")
        #[Num_wav, 1, 128]のリスト
    return features_list


    
def updateFeatures(curr_f_list, new_f):
    features_list = curr_f_list.append(new_f)
    return features_list


