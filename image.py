import cv2 as cv
from PyQt5.QtCore import QObject, pyqtSlot

class loadImage(QObject):
    def __init__(self, path_file:str):
        super().__init__(parent=None)
        self.image_path = path_file

    # @pyqtSlot()
    def processing(self):
        img = cv.imread(self.image_path)
        img_RGB = None
        img_gray = None
        if len(img.shape) == 3:
            img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            img_gray = img.copy()
            img_RGB = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        
        return {'imgRGB': img_RGB, 'imgGray': img_gray}
    