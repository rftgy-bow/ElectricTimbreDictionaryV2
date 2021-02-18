# -*- coding: utf-8 -*-
from sklearn.decomposition import PCA
#import matplotlib.pyplot as plt
import numpy as np

#主成分分析
def preparePCA(npz_path, num_components=20):
    #PCAの変換行列を算出するために
    #ESC50から抽出した特徴ベクトル50個を.npzとして保存しておき
    #ここで読み込む
    npz_data = np.load(npz_path)
    default_f = npz_data["features"]
    default_f = np.array(default_f).squeeze() #余分な次元を削除
    pca = PCA(num_components)
    pca.fit(default_f)
    return pca

def compactFeatures(features_list, pca_default):
    #listをarrayに戻す
    print("projection to PCA")
    features_array = np.array(features_list)
    if len(features_array.shape) == 1:
        #データが一つだけの場合　次元を追加
        features_array = features_array[np.newaxis, :]
    
    feature20 = pca_default.transform(features_array) #projection
    return feature20

def cosDistance(f1, f2):
    #cos = 内積 / (ノルム＊ノルム)
    d = np.dot(f1,f2) / (np.linalg.norm(f1)*np.linalg.norm(f2))
    return -d + 1

def _dimWiseDist(query, target):
    #次元別に距離を算出 
    #入力は1つの音の特徴ベクトル
    dwd = np.empty(20)
    for i in range(20):
        dwd[i] = query[i] - target[i]
    
    return dwd


def calcDistance(query, targets_list, d_type="euclid"):
    distance_list = []
    for target in targets_list:
        if d_type=="euclid":
            distance = np.linalg.norm(query-target)
        
        if d_type=="cos":
            distance = cosDistance(query, target)
            #print("cos distance")
        
        if d_type=="dimension":
            distance = _dimWiseDist(query, target)
            #print("dimension wise distance")
        
        distance_list.append(distance)
    return distance_list
