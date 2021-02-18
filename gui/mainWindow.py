# coding: utf-8
import PySimpleGUI as sg
import os
import numpy as np
import simpleaudio

from timbreSpaceWin import TimbreSpace
from module.extractor import prepareModel, extractFeatures
from module.measure import preparePCA, compactFeatures

# 初期設定
#sg.theme('DarkAmber')   # デザインテーマの設定

menu_def = [['File', ['Open', 'Save', 'Exit',]],
            ['Edit', ['Paste', ['Special', 'Normal',], 'Undo'],],
            ['Help', 'About...'],]

#列ごとのデフォルト項目
names = ["grugan.wav","bakuon_ga.wav"]
tags = ["speech", "speech"]
paths = ["F:/dataset/speech/grugan.wav", "F:/dataset/speech/bakuon_ga.wav"]

#表のデータを生成
header = ["Name", "Tag", "Path"]
table_data = [names, tags, paths]
table_data = np.array(table_data).T.tolist() #リストを行列として転置

#モデルの準備
svm_path = "./module/saved_model2"
npz_path = "./module/ESC-50_extracted_128.npz"

# ウィンドウに配置するコンポーネント
layout = [  [sg.Menu(menu_def, key="menubar")],
            [sg.Table(table_data, headings=header, select_mode="browse", max_col_width=100,
            justification="left", vertical_scroll_only=False, enable_events=True, key="MainTb")],
            [sg.FilesBrowse("Add Files", file_types=(("Audio Files", "*.wav"),), enable_events=True, key="-FILES_INPORTED-")],
            [sg.Button("Timbre Space",key="-MAKE-",size=(10,1))] ]

#################################################
# 関数

# Tableの更新
def updateTable(new_paths, current_table_data, window):
    temp_paths = new_paths.split(';')
    temp_names = []
    temp_tags = []
    for path in temp_paths:
        filename = os.path.basename(path)
        print("filename:", filename)
        temp_names.append(filename)
    
    for i in range(len(temp_paths)):
        temp_tags.append("-")

    new_items = [temp_names, temp_tags, temp_paths]
    new_items = np.array(new_items).T.tolist()
    
    new_table = np.vstack([np.array(current_table_data), np.array(new_items)]).tolist()
    window["MainTb"].Update(new_table) 
    return temp_paths


def playAudio(wav_path):
    simpleaudio.stop_all()
    wav_path = wav_path.replace(os.sep,'/')
    wav_obj = simpleaudio.WaveObject.from_wave_file(wav_path)
    play_obj = wav_obj.play()
    #play_obj.wait_done()

def makeSecondWindow(selected_path):
    window2 = TimbreSpace()
    window2.main()
    del window2

def updateF20List(file_paths, curr_f20_list):
    #モデルの読み込み
    model = prepareModel(svm_path)

    #特徴抽出(型はlistに統一)
    f_list = extractFeatures(file_paths, model)
    f_list = list(np.array(f_list).squeeze()) #余分な次元を削除

    #主成分分析
    pca_default = preparePCA(npz_path, num_components=20)
    f20_list = list(compactFeatures(f_list, pca_default))
    
    #メインのリストに追加
    curr_f20_list = curr_f20_list.append(f20_list)
    print("***Updated feature list***")
    return f20_list


#################################################

# MAINウィンドウの生成
window = sg.Window('Electric Timbre Dictionary II (v0.02)', layout, resizable=True)
selected_path = None
f20_list = [] #PCA後の特徴ベクトル

# MAINループ
while True:
    event, values = window.read()
    #print("イベント: ", event,", 値: ", values)
    if event == sg.WIN_CLOSED or values["menubar"] == "Exit":
        break
    #ファイルの追加
    elif event == '-FILES_INPORTED-':
        current_table_data = window["MainTb"].get()
        print("***current_table***",len(current_table_data), " items")
        new_paths = updateTable(values['-FILES_INPORTED-'], current_table_data, window)
        f20_list = updateF20List(new_paths, f20_list) #特徴抽出
    
    #ファイルが選択されたら再生
    elif event == 'MainTb':
        current_table_data = window["MainTb"].get()
        selected_index = values["MainTb"][0]
        selected_path = current_table_data[selected_index][2]
        print("Selected: ", selected_path)
        playAudio(selected_path)
        
    elif values["menubar"] == "Open":
        sg.popup_get_file("Open audio files")
    
    #別ウインドウを開く
    elif event == "-MAKE-" and selected_path != None:
        makeSecondWindow(selected_path) 

        
window.close()

####TO DO########################################
    
#起動時にポップアップ

#読み込みの進捗表示
#データが一個だけのときはリストに変換

#並べ替えはどうする？？？　
#名前の順番を変えたらデータの順番も変えないといけない

#右クリックメニュー
#パスを右クリックでコピーできるようにする

# リストに追加するのはQueryの分だけ？
# サーバに置いたデータを検索対象として読み込めるようにしたい

# ファイルリストと特徴量ベクトルの保存・読み込み機能

##### BUGS

#ファイル名が重複してしまう -> Setに変換　Listに戻す
#ウインドウサイズに併せてTableのサイズも変更したい

#ファイルを1つだけ追加するとエラーになってしまう

#パスが存在しないときの例外処理



