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

#################################################

def playAudio(wav_path):
    simpleaudio.stop_all()
    wav_path = wav_path.replace(os.sep,'/')
    try:
        wav_obj = simpleaudio.WaveObject.from_wave_file(wav_path)
        play_obj = wav_obj.play()
    except:
        errtxt = "Error: Cannnot play the file"
        sg.popup_ok(errtxt)


# 検索
def initSearch(q_path, q_f20, t_path, t_f20, num_result=10): #main
    #q_f20は特徴ベクトル、t_f20は特徴ベクトルの「リスト」   
    distance_list = calcDistance(q_f20, t_f20) #距離計算
    indice = np.argsort(distance_list) #データの順番を保存
    
    distance_list.sort() #距離データをソート
    distance_list = distance_list[1:num_result+1] #クエリそのものは除く
    
    #配列にキャスト
    t_path = np.array(t_path)
    t_f20 = np.array(t_f20)
    
    sorted_t_paths = t_path[indice] #距離順にパスのリストをソート
    sorted_t_paths = sorted_t_paths[1:num_result+1] 
    
    sorted_t_f20 = t_f20[indice] #特徴ベクトルもソート
    sorted_t_f20 = sorted_t_f20[1:num_result+1] 
    
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


# GUIがぼやける現象を防ぐための関数
def make_dpi_aware():
  import ctypes
  import platform
  if int(platform.release()) >= 8:
    ctypes.windll.shcore.SetProcessDpiAwareness(True)


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

class TimbreSpace(): 

    #初期設定
    def __init__(self, query_path, query, targets_path, targets):
        self.query = query
        self.query_path = query_path
        self.targets = targets
        self.targets_path = targets_path
        
        #類似度高い順に10件取得
        self.result_f20, self.result_paths, self.distance_list = \
            initSearch(query_path, query, targets_path, targets)
        
        # レイアウト作成
        make_dpi_aware()
        sns.set_style("darkgrid")
        table_data = makeTableData(self.distance_list, self.result_paths)
        header = ["Filename", "Distance"]
        top_text = "Query: " + os.path.basename(query_path)
        
        layout = [[sg.Text(top_text, enable_events=True, key="topText")],
                  [sg.Canvas(key='-CANVAS-'),
                  sg.Table(table_data, headings=header, select_mode="browse", 
                  enable_events=True, key="resultTb")],
                  [sg.Button("↑", key="up")],
                  [sg.Button("←", key="left"), sg.Button("→", key="right")],
                  [sg.Button("↓", key="down")] ]
        
        # windowを作成する．finalize=Trueにする必要がある．
        self.window = sg.Window('Electric Timbre Dictionary v2.00b', layout, finalize=True, 
            element_justification='center', font='Monospace 18',
            return_keyboard_events=True, use_default_focus=False)
        
        
        # 埋め込み用figを作成する
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        makePoints(self.result_f20, query, self.ax)
        
        # figとCanvasを関連付ける．
        self.fig_agg = draw_figure(self.window['-CANVAS-'].TKCanvas, self.fig)
    


    #################################################

    # 起動後にユーザの操作で呼び出される関数
        
    def refleshAx(self, coords, ax):
        return 0


    def rotateAx(self, ax, direction, fig_agg):
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
    def selectItem(self, selected_index, result_paths):
        selected_path = result_paths[selected_index]
        print("Selected: ", selected_path)
        playAudio(selected_path)


    # 選んだ音の表示色を変える
    def selectPoints(self, selected_index, result_f20, ax, fig_agg):
        coords = result_f20[selected_index]
        x = coords[0]
        y = coords[1]
        z = coords[2]
        ax.scatter(x,y,z, s=200, c="orange", edgecolors="red")
        fig_agg.draw()


        
    #################################################


    def main(self):
        arrow_events = ["left", "right", "up", "down",
                "Left:37", "Right:39", "Up:38", "Down:40"]
        
        # イベントループ
        # 分岐後に呼びだす関数名のみを書くこと
        while True:
            event, values = self.window.read()
            print(event, values)
            # sg.Print(event, values)

            if event in (None, "Cancel"):
                break
             
            elif event in arrow_events:
                self.rotateAx(self.ax, event, self.fig_agg)
                
            #ファイルが選択されたら再生
            elif event == "resultTb":
                self.selectItem(values["resultTb"][0], self.result_paths)
                self.selectPoints(values["resultTb"][0], self.result_f20, 
                                  self.ax, self.fig_agg)
                
            elif event == "topText":
                playAudio(self.query_path)
            
                
        # ウィンドウを閉じる．
        self.window.close()
    
#################################################

#処理の流れ
#クエリのパスQPと特徴量QFを受け取る

#QFを使って音検索(100件)
#上位10件をリストに追加

#上位10件が収まるように次元を選ぶ
#10個の音＋クエリを空間に表示

#リスト上のアイテムが選ばれたら該当する点をオレンジ色に

