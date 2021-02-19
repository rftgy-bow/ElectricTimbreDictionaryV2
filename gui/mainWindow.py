# coding: utf-8
import PySimpleGUI as sg
import os
import numpy as np
import simpleaudio

from timbreSpaceWin2 import TimbreSpace
from module.extractor import prepareModel, extractFeatures
from module.measure import preparePCA, compactFeatures

# 初期設定
#sg.theme('DarkAmber')   # デザインテーマの設定

menu_def = [['File', ['Open', 'Save', 'Exit',]],
            ['Edit', ['Paste', ['Special', 'Normal',], 'Undo'],],
            ['Help', 'About...'],]

#列ごとのデフォルト項目
names = ["sample.wav"]
paths = ["./sample.wav"]

#表のデータを生成
header = ["Name", "Path"]
table_data = [names, paths]
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
def updateTable(new_paths, current_table_data, window, load=False):
    if load == False:
        temp_paths = new_paths.split(';')
    else:
        temp_paths = new_paths
    
    temp_names = []
    for path in temp_paths:
        filename = os.path.basename(path)
        print("filename:", filename)
        temp_names.append(filename)
       
    new_items = [temp_names, temp_paths]
    new_items = np.array(new_items).T.tolist()
    
    new_table = np.vstack([np.array(current_table_data), np.array(new_items)]).tolist()
    window["MainTb"].Update(new_table)
    
    curr_paths = np.array(new_table)[:,1].tolist()
    return curr_paths, temp_paths


def playAudio(wav_path):
    simpleaudio.stop_all()
    wav_path = wav_path.replace(os.sep,'/')
    try:
        wav_obj = simpleaudio.WaveObject.from_wave_file(wav_path)
        play_obj = wav_obj.play()
    except:
        errtxt = "Error: Cannnot play the file"
        sg.popup_ok(errtxt)
        
    #play_obj.wait_done()


def updateF20List(file_paths, curr_f20s):
    #モデルの読み込み
    model = prepareModel(svm_path)

    #特徴抽出(型はlistに統一)
    f_list = extractFeatures(file_paths, model)
    f_list = list(np.array(f_list).squeeze()) #余分な次元を削除

    #主成分分析
    pca_default = preparePCA(npz_path, num_components=20)
    f20_list = list(compactFeatures(f_list, pca_default))
    
    #メインのリストに追加
    for f20 in f20_list:
        #リストを開いて1個ずつ追加
        curr_f20s.append(f20)
    
    print("***Updated feature list***")
    return curr_f20s


def open_timbre_list(open_path, curr_paths, curr_f20s):
    paths = np.load(open_path)["paths"]
    f20s = np.load(open_path)["features"]
    curr_f20s.append(f20s)
    return paths, curr_f20s

def save_timbre_list(save_path, curr_paths, curr_f20s):
    np.savez(save_path, paths=curr_paths, features=curr_f20s)
    print("saved timbre list:", save_path)


def makeSecondWindow(selected_path, selected_index, curr_paths, curr_f20s):
    #curr_pathsの0番目はダミー
    query_path = selected_path
    query = curr_f20s[selected_index]
    targets_path = curr_paths
    targets = curr_f20s
    
    window2 = TimbreSpace(query_path, query, targets_path, targets)
    window2.main()
    del window2


#################################################

# MAINウィンドウの生成
selected_path = None
curr_paths = paths
curr_f20s = [] #PCA後の特徴ベクトル
curr_f20s.append(np.zeros(20))

window = sg.Window('Electric Timbre Dictionary II (v0.02)', layout, resizable=True)

# MAINループ
while True:
    event, values = window.read()
    #print("イベント: ", event,", 値: ", values)
    if event == sg.WIN_CLOSED or values["menubar"] == "Exit":
        break
    #ファイルの追加
    elif event == '-FILES_INPORTED-' and values['-FILES_INPORTED-'] != "":
        current_table_data = window["MainTb"].get()
        print("***current_table***", len(current_table_data), " items")
        curr_paths, new_paths =\
            updateTable(values['-FILES_INPORTED-'], current_table_data, window)
        curr_f20s = updateF20List(new_paths, curr_f20s) #特徴抽出
    
    #ファイルが選択されたら再生
    elif event == 'MainTb':
        current_table_data = window["MainTb"].get()
        selected_index = values["MainTb"][0]
        selected_path = current_table_data[selected_index][1]
        print("Selected: ", selected_index, selected_path)
        playAudio(selected_path)
        
    elif values["menubar"] == "Open":
        list_open_path = sg.popup_get_file("Load timbre list (audio paths and features)", 
            file_types=(("numpy binary zip", "*.npz"),))
        
        if list_open_path not in ["", None]:
            current_table_data = window["MainTb"].get()
            new_paths, curr_f20s = open_timbre_list(list_open_path, curr_paths, curr_f20s)
            updateTable(new_paths, current_table_data, window, load=True)
    
        
    elif values["menubar"] == "Save":
        list_save_path = sg.popup_get_file("Save timbre list (audio paths and features)", 
            file_types=(("numpy binary zip", "*.npz"),), save_as=True)
        
        if list_save_path not in ["", None]:
            save_timbre_list(list_save_path, curr_paths, curr_f20s)
    
    #別ウインドウを開く
    elif event == "-MAKE-" and selected_path != None:
        makeSecondWindow(selected_path, selected_index, curr_paths, curr_f20s)

        
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



