import sys
import os
import time
import typing
from PyQt5.QtCore import QObject
from numpy import argmin
from libraries.ComboWidget import ComboWidget
from libraries.display.display import Display
from libraries.display.shape import Shape
from libraries.dialog import textDialog, cprogressDialog
from functools import partial
from image import loadImage
# from yolov6.core.inferer_custom import CustomInferer
from yolo_onnx.inference import OnnxInference
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PyQt5 import QtGui, QtWidgets, QtCore
__appname__ = "Detection"
# Thay label cua model cua ban
LABEL_NAMES = ["line", "scope", "title", "error", "background"]
id_color_nodule = [(102,205,170, 255),
                    (0,250,154, 255),
                    (255, 156, 0, 255), 
                    (255, 0, 0, 255),
                    (54,156,223,255)]
id_color_nodule_fill = [(102,205,170, 125),
                        (0,250,154, 125),
                        (255, 156, 0, 125), 
                        (255, 0, 0, 125), 
                        (54, 156, 223, 125)]
def new_icon(icon):
    return QtGui.QIcon("./sources/" + icon)

def new_action(parent, text, slot=None, shortcut=None, icon=None,
               tip=None, checkable=False, enabled=True, trigger= False):
    """Create a new action and assign callbacks, shortcuts, etc."""
    a = QtWidgets.QAction(text, parent)
    if icon is not None:
        a.setIcon(new_icon(icon))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts(shortcut)
        else:
            a.setShortcut(shortcut)
    if tip is not None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    if slot is not None:
        a.triggered.connect(slot)
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enabled)
    return a

def add_actions(widget, actions):
    for action in actions:
        if action is None: 
            widget.addSeparator()
        # elif isinstance(action, QWidget):
        #     widget.addWidget(action)
            # print("add Widget")
        elif isinstance(action, QtWidgets.QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)
            # print("add action")

def getFileDefault():
    import yaml
    file_default = os.path.join(os.getcwd(),"default.yaml")
    with open(file_default, 'r') as file:
        configuration = yaml.safe_load(file)
        default_path = os.path.join(os.getcwd(),configuration['default_model'])
        load_image_path = configuration['folder_path'] 
        if not os.path.exists(load_image_path):
            load_image_path = ""
        file.close()
        return default_path, load_image_path, configuration

def saveFileDefault(file_default, configuration):
    import yaml
    with open(file_default, 'w+') as file:
        yaml.dump(configuration, file)

class Ui_MainWindow(object):
    
    def controlLayout(self, use_gpu:bool):
        folder_symbol = '\U0001f4c2'  # 📂
        model_default, self.default_image_path, self.configuration = getFileDefault()
        self.open_file = QtWidgets.QPushButton(folder_symbol + "  Open", self)
        self.list_model_name = ComboWidget(self, "Model", model_default, use_gpu)

        self.show_label = QtWidgets.QCheckBox("Show Label", self)
        self.show_label.setChecked(True)

        self.btn_run = QtWidgets.QPushButton("Run", parent=self)
        self.btn_run.setEnabled(False)
        widget = QtWidgets.QWidget(self)
        control_layout = QtWidgets.QHBoxLayout(widget)
        control_layout.addWidget(self.open_file)
        control_layout.addWidget(self.list_model_name, 1)
        control_layout.addWidget(self.show_label)
        control_layout.addStretch(1)
        control_layout.addWidget(self.btn_run)
        return widget
    
    def display(self):
        widget = QtWidgets.QWidget(self)
        self.original = Display()
        self.original.canvas.pixmap = None

        self.result_display = Display()
        self.result_display.canvas.pixmap = None
        self.result_display.setEnabled(False)
        layout = QtWidgets.QHBoxLayout(widget)
        layout.addWidget(self.original)
        layout.addWidget(self.result_display)
        return widget


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.startParam()
        self._setupUi()

    def startParam(self):
        self.image_data = None
        self.image_gray = None
        self.result = None
        self.use_gpu = True
        self.device = "0"
        self.default_image_path = ""
        # self.model = CustomInferer(img_size=640, half=True)
        self.model = OnnxInference() #Thay cai thanh cai model cua em
        # self.inferModel = inferenceThread(None, self.model)
        
        pass
        
    def _setupUi(self):
        # Create the main window and set its properties
        self.setObjectName("MainWindow")
        # self.setStyleSheet(open(r"libraries\pyqt5-dark-theme.stylesheet").read())
        self.setStyleSheet(open(r"libraries\style_DarkOrange.qss").read())
        # self.setStyleSheet(open(r"libraries\style_navy.css").read())
        
        self.centralWidget = QtWidgets.QWidget(self)
        self.centralWidget.setObjectName("centralWidget")

        # Main Window
        window_layout = QtWidgets.QVBoxLayout(self.centralWidget)
        # List Model and Button
        self.control_widget = self.controlLayout(self.use_gpu)
        self.display_widget = self.display()
        window_layout.addWidget(self.control_widget)
        window_layout.addWidget(self.display_widget)
        self.setCentralWidget(self.centralWidget)

        QtCore.QMetaObject.connectSlotsByName(self)
        # set connect ui
        self.open_file.clicked.connect(self.selectImage)
        self.btn_run.clicked.connect(self.runModel)
        self.original.drop_file.connect(self.load_image)
        self.original.isImage.connect(self.btn_run.setEnabled)
        self.list_model_name.selectItem.connect(self.initModel)
        self.list_model_name.use_gpu.stateChanged.connect(self.changeStatusCheckBox)
        self.show_label.stateChanged.connect(self.setLabel)

        #initial function
        print(self.list_model_name.getCurrentText())
        self.initModel(self.list_model_name.getCurrentText())

    # Initial Action
    def _setAction(self):
        action = partial(new_action, self)
        open = action('Open File', self.selectImage,
                      'Ctrl+O', 'open', 'open File Detail') 
        # add_actions(QtWidgets.QMenu, open)
        zoom = QtWidgets.QWidgetAction(self)
        zoom.setDefaultWidget(self.display.zoomWidget)
        # self.display.zoomWidget.setWhatsThis(u"Zoom in/out of the image.  Also accessible with " "%s and %s from the canvas." % (format_shortcut("Ctrl+[-+]"), format_shortcut("Ctrl+Wheel")))

        zoom_in = action('Zoom In', partial(self.result_display.add_zoom, 10),
                         'Ctrl++', 'zoom-in', 'zoominDetail', enabled=False)
        zoom_out = action('Zoom Out', partial(self.result_display.add_zoom, -10),
                          'Ctrl+-', 'zoom-out', 'zoomoutDetail', enabled=False)
        
    def load_image(self, image_path):
        image = loadImage(path_file=image_path)
        image_dict = image.processing()
        
        self.image_data = image_dict["imgRGB"]
        self.image_gray = image_dict["imgGray"]
        self.original.load_image(self.image_data)
        # self.original.scale_fit_width()
        self.result_display.load_image(self.image_data)
        pass

    def selectImage(self):
        filedialog=QtWidgets.QFileDialog()
        filename,_=filedialog.getOpenFileName(self, "Open Image File", self.default_image_path , filter="Images (*.png *.jpg *.bmp)")
        if not len(filename)==0:
            print(filename)
            self.load_image(filename)
            dir = os.path.dirname(filename)
            if dir != self.default_image_path:
                self.default_image_path = dir
                self.configuration['folder_path'] = dir
                saveFileDefault(os.path.join(os.getcwd(),"default.yaml"), self.configuration)
                

    def errorMessage(self, title, message):
        """
        Show Warning Dialog
        """
        return QtWidgets.QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))
    
    def initModel(self, modelname):
        # Load model from pt file in threading
        self.device = "0" if self.use_gpu else "cpu"
        if isinstance(modelname, str):
            try:
                textLoading = textDialog(self)
                textLoading.show()
                QtWidgets.QApplication.processEvents()
                
                objThreading = QtCore.QThread()
                inferModel = inferenceThread(None, self.model)
                inferModel.stringOutSignal.connect(textLoading.setText)
                inferModel.moveToThread(objThreading)
                inferModel.finishedSignal.connect(objThreading.quit)
                objThreading.started.connect(partial(inferModel.initModel, modelname, self.device,))
                objThreading.finished.connect(textLoading.close)
                objThreading.start()
                while objThreading.isRunning():
                    self.setEnabled(False)
                    QtWidgets.QApplication.processEvents()
                self.setEnabled(True)
                del objThreading
            except Exception as e:
                msg = "Error loading the model" + "\n\t"  \
                "{}".format(e.__str__())
                self.errorMessage("",msg )
            self.result_display.update_shape([])

    def changeStatusCheckBox(self):
        if self.list_model_name.use_gpu.isChecked():
            self.use_gpu = True
            self.device = "0"
        else:
            self.use_gpu = False
            self.device = 'cpu'

    def showResult(self, results:dict):
        # print(results)
        self.result_display.setEnabled(True)
        s = []
        for (key, value) in results.items():
            label = LABEL_NAMES[key]
            for *xyxy, conf in value:
                points = ((xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]))
                shape = self.setShape(label,
                                      points,
                                      "rectangle",
                                      category=key,
                                      conf = (conf*100), paint_label=True)
                s.append(shape)
        self.result_display.update_shape(s)
        # self.result_display.canvas.mode = self.result_display.canvas.EDIT
    
    def runModel(self):
        try:
            image = self.image_gray
            print(self.model.data_lengh())
            progressDialog = cprogressDialog("Run Inference ...", "", 0, self.model.data_lengh(), self)
            progressDialog.show()
            QtWidgets.QApplication.processEvents()
            objThreading2 = QtCore.QThread()
            inferModel = inferenceThread(None, self.model)
            inferModel.progressSignal.connect(progressDialog.setValue)
            inferModel.resultSignal.connect(self.showResult)
            inferModel.moveToThread(objThreading2)
            inferModel.finishedSignal.connect(objThreading2.quit)
            objThreading2.started.connect(partial(inferModel.infer, image, 0.3, 0.65,))
            objThreading2.finished.connect(progressDialog.canceled)
            objThreading2.start()
            while objThreading2.isRunning():
                self.setEnabled(False)
                QtWidgets.QApplication.processEvents()
            self.setEnabled(True)
            del objThreading2
        except Exception as e:
            msg = "Error Run Inference" + "\n\t"  \
            "{}".format(e.__str__())
            self.errorMessage("",msg )
        pass
    
    def setShape(self, label, points, shape_type, category, conf, paint_label=True):
        shape = Shape(label=label,
                      shape_type=shape_type,
                      conf=conf,
                      paint_label=paint_label)
        for x, y in points:
            x, y, snapped = self.result_display.canvas.snapPointToCanvas(x, y)
            if snapped: self.set_dirty()
            shape.addPoint(QtCore.QPointF(x, y))
        shape.close()
        line_color = id_color_nodule[category]
        fill_color = id_color_nodule_fill[category]
        if line_color:
            shape.line_color = QtGui.QColor(*line_color)
        else:
            shape.line_color = QtGui.QColor('cyan')
        shape.line_color = QtGui.QColor(*line_color) if line_color else QtGui.QColor('cyan')
        shape.fill_color = QtGui.QColor(*fill_color) if line_color else QtGui.QColor('blue')
        return shape
    
    def setLabel(self):
        self.result_display.showLabel(self.show_label.isChecked())

    def set_dirty(self):
        pass
        

class inferenceThread(QtCore.QObject): # Thay inference cua model cua ban
    finishedSignal = QtCore.pyqtSignal()
    stringOutSignal = QtCore.pyqtSignal(str)
    progressSignal = QtCore.pyqtSignal(int)
    resultSignal = QtCore.pyqtSignal(dict)

    def __init__(self, parent: None, function:OnnxInference) -> None:
        super().__init__(parent)
        self._function = function

    @QtCore.pyqtSlot()
    def initModel(self, modelname, device):
        self.stringOutSignal.emit("Loading checkpoint from {}".format(modelname))
        self.stringOutSignal.emit("Fusing model...")
        self._function.initModel(modelname, device)
        self.stringOutSignal.emit("Initial model DONE")       
        self.finishedSignal.emit()

    @QtCore.pyqtSlot()
    def infer(self,
              image, 
              conf_thresh, 
              iou_thresh, 
              classes = None, 
              agnostic_nms = False, 
              max_det= 1000,):
        self._function.data.loadImage(image)
        results = {}
        for index, (img_src, pos, ) in enumerate(tqdm(self._function.data)):
            pos, result = self._function.infer_for_step(img_src, pos, 
                                                        conf_thresh,
                                                        iou_thresh, 
                                                        classes,
                                                        agnostic_nms ,max_det,)
            results[pos] = result
            self.progressSignal.emit(index+1)
        mergeResults = self._function.mergeBBoxOffset(results)
        mergeResults = self._function.postProcessing(mergeResults)
        self.resultSignal.emit(mergeResults)
        self.finishedSignal.emit()


        

if __name__ == "__main__":
    full_path = os.path.realpath(__file__)
    os.chdir(os.path.dirname(full_path))
    app = QtWidgets.QApplication(sys.argv)
    # app.setStyle('QtCurve')
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())