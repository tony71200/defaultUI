from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QComboBox, QLabel, QWidget, QToolButton, QCheckBox
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import pyqtSignal
from glob import glob
import os

def checkWeightPath():
    if not os.path.exists('./weights/'):
        raise Exception("Weights folder doesnot exist!")
    elif (os.listdir("./weights/")==[]):
        raise Exception("No weights found in./weights/. Please download from https://github.com/ultralytics/yolov6#pretrained-checkpoints or train your own model first.")
    else:
        model_list = glob(os.path.join(os.getcwd(), 'weights/*.onnx'))
    return model_list

class ComboWidget(QWidget):
    refresh_symbol = "\U0001F504"  # 🔄
    selectItem = pyqtSignal(str)
    def __init__(self,parent, name:str, text_default:str, use_gpu:bool) -> None:
        super().__init__(parent=parent)
        self.modelname = None
        self.index = 0
        self.model_default = text_default
        self._combo = QComboBox(self)
        self.model_list = checkWeightPath()
        self.setList(self.model_list)
        self.setTextDefault(self.model_default)
        _label = QLabel(name, self)
        # self._label.setSizePolicy(QSizePolicy.Prefered, QSizePolicy.Prefered)
        _label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        self.btn_refresh = QToolButton()
        self.btn_refresh.setText(ComboWidget.refresh_symbol)
        self.use_gpu = QCheckBox("GPU", self)
        self.use_gpu.setChecked(use_gpu)
        self.use_gpu.setVisible(False)

        layout = QtWidgets.QHBoxLayout(self)

        layout.addWidget(_label)
        layout.addWidget(self._combo, 1)
        layout.addWidget(self.btn_refresh)
        layout.addWidget(self.use_gpu)
        layout.setContentsMargins(5,0,5,0)

        # self._combo.currentTextChanged.connect(self.changeSelection)
        self._combo.activated.connect(self.comActivated)
        # self._combo.activated.connect(self.changeSelection)
        self.btn_refresh.clicked.connect(self.refreshModelList)
     
    def changeSelection(self, text):
        if (text in self.model_list):
            self.selectItem.emit(text)
            self.index = self.model_list.index(text)
            self.modelname = text
        else:
            self.index = 0
            self.modelname = self.model_list[self.index]
            self._combo.setCurrentText(self.modelname)
            self.selectItem.emit(self.modelname)
        pass

    def comActivated(self, index):
        # print("Activated index:", index)
        text = self.model_list[index]
        if (text in self.model_list):
            self.selectItem.emit(text)
            self.index = self.model_list.index(text)
            self.modelname = text
        else:
            self.index = 0
            self.modelname = self.model_list[self.index]
            self._combo.setCurrentText(self.modelname)
            self.selectItem.emit(self.modelname)
        pass

    def setList(self, combo_list:list):
        self._combo.addItems(combo_list)
        try:
            self.modelname = combo_list[self.index]
        except:
            #             print('no model selected')
            if len(combo_list) >0:
                self.modelname = combo_list[-1]
                self.selectItem.emit(self.modelname)
            else: 
                print('no model selected')

    def getCurrentText(self):
        return self._combo.currentText()

    def setCheck(self, checked):
        self.use_gpu.setChecked(checked)

    def getGPUChecked(self):
        return bool(self.use_gpu.isChecked())
    
    def refreshModelList(self):
        self.model_list = checkWeightPath()
        self._combo.clear()
        self.setList(self.model_list)
        print("refresh button:", self.modelname)
        # self.setTextDefault(r"D:\002_code\scope-UI_3\weights\scope6m_cbam_230817.pt")
        if self.modelname in self.model_list:
            self.index=self.model_list.index(self.modelname)
            self._combo.setCurrentIndex(self.index)
        else:
            print('Warning! Model not found!')
            self.setTextDefault(self.model_default)
            self.selectItem.emit(self.modelname)

    def setTextDefault(self, textDefault):
        if textDefault in self.model_list:
            self.index=self.model_list.index(textDefault)
            self._combo.setCurrentIndex(self.index)
            self.modelname = textDefault



    

    

    

    



