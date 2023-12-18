from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt
import sys

class textDialog(QtWidgets.QDialog):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.__setupUI()
        self.string = ""

    def __setupUI(self):
        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowTitleHint| Qt.WindowSystemMenuHint & ~Qt.WindowCloseButtonHint)
        self.setWindowTitle("Loading ...")
        self.message = QtWidgets.QLabel(self)
        self.message.setWordWrap(True)
        layout=QtWidgets.QVBoxLayout()
        layout.addWidget(self.message)
        self.setLayout(layout)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)

    def setText(self, text):
        self.string += text +'\n'
        self.message.setText(self.string)

    def close(self) -> bool:
        self.string = ""
        return super().close()

class cprogressDialog(QtWidgets.QProgressDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set the window title to "Processing" and disable close button hint so that we can still cancel it with ESC key press event handle
        # Disable close button
        self.setWindowFlags(
            (
            self.windowFlags() | Qt.CustomizeWindowHint  # type: ignore[operator]
            ) & ~Qt.WindowContextHelpButtonHint   # type: ignore[operator]
            )
        
        self.setLabelText("Run Inference ...")
        self.setWindowTitle("Run Inference ...")
        self.setCancelButton(None)
        self.autoClose()
        self.autoReset()
        self.reset()
    

