# coding: utf-8

print("importing modules...")
from module.extractor import prepareModel, extractFeatures
from module.measure import preparePCA, compactFeatures, calcDistance
import glob
import numpy as np
import time

svm_path = "./module/saved_model2"
npz_path = "./module/ESC-50_extracted_128.npz"

#audio_dir = "F:/SoundIdeas"
#query_dir = "F:/dataset/osaka_sound/効果音課題2021.1.1"
#audio_dir = "./test_wav"
audio_dir="F:/Users/KOMIHIRO/Music/SE\MAD用音声素材/逆転裁判/se/wav"

def dirToF20(audio_dir, svm_path, npz_path):
    #ファイルパスの収集
    target = audio_dir + "/**/*.wav"
    file_paths = glob.glob(target, recursive=True)
    
    #モデルの読み込み
    model = prepareModel(svm_path)
    
    #データが1個(str)の場合　無理矢理listに変換
    if type(file_paths) is not list:
        temp_path = file_paths
        file_paths = []
        file_paths.append(temp_path)
    
    #特徴抽出(型はlistに統一)
    f_list = extractFeatures(file_paths, model)
    f_list = list(np.array(f_list).squeeze()) ##余分な次元を削除
    
    #主成分分析
    pca_default = preparePCA(npz_path, num_components=20)
    f20_list = list(compactFeatures(f_list, pca_default))
    return file_paths, f20_list

#################################################

# 時間計測開始
time_sta = time.time()
print("***process start***")

all_sound_paths, all_sound_f20 = dirToF20(audio_dir, svm_path, npz_path) 
filename = "gyakuten_saiban_f20.npz"

np.savez(filename, paths=all_sound_paths, features=all_sound_f20)

#filename = "osaka_sound_features20_query.npz"
#query_paths, query_f20 = dirToF20(query_dir, svm_path, npz_path)
#np.savez(filename, paths=query_paths, features=query_f20)

# 時間計測終了
time_end = time.time()
# 経過時間（秒）
tim = time_end - time_sta

print("***exec time:", tim, "sec***")