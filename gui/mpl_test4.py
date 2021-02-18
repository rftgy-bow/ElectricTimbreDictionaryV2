#matplotlibで3Dグラフを描画する
#https://white-wheels.hatenadiary.org/entry/20100327/p3

# 3Dグラフの埋め込み
print("Now loading...")
import numpy as np
import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from module.measure import calcDistance, cosDistance
import os
import simpleaudio
sns.set_style("darkgrid")

#################################################

# デモ用
#q_npz_path = "./npz/test_wav_f20.npz"
q_npz_path = "./npz/gyakuten_saiban_f20.npz"
#q_npz_path = "./npz/osaka_sound_features20_query.npz"
queries_path = np.load(q_npz_path)["paths"]
queries_list = np.load(q_npz_path)["features"]

index = 41

query_path = queries_path[index-1]
query = queries_list[index-1] 

t1_npz_path = "./npz/series6000_f20.npz"
targets_path = np.load(t1_npz_path)["paths"]
targets_list = np.load(t1_npz_path)["features"]

#Extentionの分のパスと特徴量を追加
t2_npz_path = "./npz/series6000_extention_f20.npz"
targets_path = np.concatenate([
    targets_path, np.load(t2_npz_path)["paths"]
    ], 0)
targets_list = np.concatenate([
    targets_list, np.load(t2_npz_path)["features"]
    ], 0)



#################################################

def playAudio(wav_path):
    simpleaudio.stop_all()
    wav_path = wav_path.replace(os.sep,'/')
    wav_obj = simpleaudio.WaveObject.from_wave_file(wav_path)
    play_obj = wav_obj.play()


# 検索
def initSearch(q_path, q_f20, t_path, t_f20, num_result=10): #main
    #q_f20は特徴ベクトル、t_f20は特徴ベクトルのリスト  
    distance_list = calcDistance(q_f20, t_f20, d_type="cos") #距離計算
    indice = np.argsort(distance_list) #データの順番を保存
    
    distance_list.sort() #距離データをソート
    distance_list = distance_list[0:num_result]
    
    sorted_t_paths = t_path[indice] #距離順にパスのリストをソート
    sorted_t_paths = sorted_t_paths[0:num_result]
    
    sorted_t_f20 = t_f20[indice] #特徴ベクトルもソート
    sorted_t_f20 = sorted_t_f20[0:num_result] 
    
    #上位10件のみを返す
    return sorted_t_f20, sorted_t_paths, distance_list


def makeTableData(distance_list, result_paths):
    result_filenames = []
    for path in result_paths:
        filename = os.path.basename(path)
        result_filenames.append(filename)
            
    table_data = [result_filenames, distance_list]
    table_data = np.array(table_data).T.tolist() #リストを行列として転置
    return table_data


#################################################
# 表示部の関数

# GUIがぼやける現象を防ぐための関数
def make_dpi_aware():
  import ctypes
  import platform
  if int(platform.release()) >= 8:
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
make_dpi_aware()

# 描画用の関数
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


# 特徴ベクトルを空間に表示
def makePoints(f20_list, query, ax):
    #shape = 10x20 (検索結果x特徴量の次元)
    f20_array = np.array(f20_list)
    
    x = f20_array[:,0]
    y = f20_array[:,1]
    z = f20_array[:,2]
    ax.scatter3D(np.ravel(x),np.ravel(y),np.ravel(z),s=100)
    
    qx = query[0]
    qy = query[1]
    qz = query[2]
    # クエリは大きく表示
    ax.scatter(qx,qy,qz, s=500, c="orange", edgecolors="red")


#################################################



#類似度高い順に10件取得
result_f20, result_paths, distance_list = \
    initSearch(query_path, query, targets_path, targets_list)


# レイアウト作成
table_data = makeTableData(distance_list, result_paths)
header = ["Filename", "Distance"]
top_text = "Query: " + os.path.basename(query_path)

layout = [[sg.Text(top_text, enable_events=True, key="topText")],
          [sg.Canvas(key='-CANVAS-'),
          sg.Table(table_data, headings=header, select_mode="browse", enable_events=True, key="resultTb")],
          [sg.Button("↑", key="up")],
          [sg.Button("←", key="left"), sg.Button("→", key="right")],
          [sg.Button("↓", key="down")] ]

# windowを作成する．finalize=Trueにする必要がある．
window = sg.Window('Electric Timbre Dictionary v2.00b', layout, finalize=True, 
    element_justification='center', font='Monospace 18',
    return_keyboard_events=True, use_default_focus=False)


# 埋め込み用figを作成する
fig = plt.figure()
ax = Axes3D(fig)
makePoints(result_f20, query, ax)

# figとCanvasを関連付ける．
fig_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)


#################################################

# 起動後にユーザの操作で呼び出される関数
    
def refleshAx(coords, ax):
    return 0


def rotateAx(ax, direction):
    c_elev = vars(ax)["elev"]
    c_azim = vars(ax)["azim"]
    if direction=="left" or direction=="Left:37":
        ax.view_init(elev=c_elev, azim=c_azim-5)
        
    elif direction=="right" or direction=="Right:39":
        ax.view_init(elev=c_elev, azim=c_azim+5)
        
    elif direction=="up" or direction=="Up:38":
        ax.view_init(elev=c_elev+5, azim=c_azim)
        
    elif direction=="down" or direction=="Down:40":
        ax.view_init(elev=c_elev-5, azim=c_azim)
        
    fig_agg.draw()
 
 
# 結果のリストから音を選んだとき
def selectItem(selected_index, result_paths):
    selected_path = result_paths[selected_index]
    print("Selected: ", selected_path)
    playAudio(selected_path)


# 選んだ音の表示色を変える
def selectPoints(selected_index, ax):
    coords = result_f20[selected_index]
    x = coords[0]
    y = coords[1]
    z = coords[2]
    ax.scatter(x,y,z, s=200, c="orange", edgecolors="red")
    fig_agg.draw()


    
#################################################

# イベントループ
# 分岐後に呼びだす関数名のみを書くこと

arrow_events = ["left", "right", "up", "down",
        "Left:37", "Right:39", "Up:38", "Down:40"]

while True:
    event, values = window.read()
    print(event, values)
    # sg.Print(event, values)

    if event in (None, "Cancel"):
        break
     
    elif event in arrow_events:
        rotateAx(ax, event)
        
    #ファイルが選択されたら再生
    elif event == "resultTb":
        selectItem(values["resultTb"][0], result_paths)
        selectPoints(values["resultTb"][0], ax)
        
    elif event == "topText":
        playAudio(query_path)
    

        
# ウィンドウを閉じる．
window.close()

#処理の流れ
#クエリのパスQPと特徴量QFを受け取る

#QFを使って音検索(100件)
#上位10件をリストに追加

#上位10件が収まるように次元を選ぶ
#10個の音＋クエリを空間に表示

#リスト上のアイテムが選ばれたら該当する点をオレンジ色に

#stringの分割を使ってパスからファイル名だけを取り出す？
