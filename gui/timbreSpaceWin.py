# coding: utf-8
# from https://qiita.com/melka-blue/items/33c89a62c2392bbbd450
import PySimpleGUI as sg

class TimbreSpace(): 
    #query_f20, targets_f20
    def __init__(self):
        sg.theme("DarkBlue11")
        self.layout = [[sg.Text("ここにmatplotlibで音色空間を表示する")],
                   [sg.Button("Exit",key="-EXIT-",size=(10,1))]]
        #keep_on_top=Trueにする
        self.window = sg.Window("Timbre Space View",self.layout,keep_on_top=True)

    def main(self):
        while True:
            event, value = self.window.read()
            if event == "-EXIT-":
                break
        self.window.close()
