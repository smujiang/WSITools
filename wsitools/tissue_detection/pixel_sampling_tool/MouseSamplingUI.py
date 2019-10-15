import sys,os
import openslide
import math
import numpy as np
from skimage import io
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5 import QtCore
from PyQt5.QtCore import QDir,Qt,pyqtSlot,QCoreApplication,QMetaObject, QObject, QPoint,QEvent
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap, QPicture, QCursor, QPen
from PyQt5.QtWidgets import (QAction,QVBoxLayout, QHBoxLayout,QPushButton,QCheckBox,QWidget,QApplication, QFileDialog, QLabel,
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy, QTextEdit, QLineEdit, QLayout, QComboBox, QSpinBox,QDoubleSpinBox )

Image.MAX_IMAGE_PIXELS = None

class StainPenMarking(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.OpenImgButton = QPushButton("Open")  # load image
        self.EditImgName = QLineEdit()  # edit line to show image name
        self.ShowImgLabel = QLabel()
        self.ShowImgLabel.installEventFilter(self)
        self.EditSaveTSV = QLineEdit()  # edit line to save tsv file
        self.SegButton = QPushButton("Seg")

        self.vbox_main = QVBoxLayout()
        self.row1_hbox = QHBoxLayout()
        self.row2_hbox = QHBoxLayout()
        self.row3_hbox = QHBoxLayout()
        self.row1_hbox.addWidget(self.EditImgName)
        self.row1_hbox.addWidget(self.OpenImgButton)
        self.row2_hbox.addWidget(self.EditSaveTSV)
        self.row2_hbox.addWidget(self.SegButton)
        self.row3_hbox.addWidget(self.ShowImgLabel)
        self.vbox_main.addLayout(self.row1_hbox)
        self.vbox_main.addLayout(self.row2_hbox)
        self.vbox_main.addLayout(self.row3_hbox)
        self.vbox_main.setSizeConstraint(QLayout.SetMinimumSize)
        self.setLayout(self.vbox_main)

        init_pixmap_img = QPixmap(800, 800)
        init_pixmap_img.fill(Qt.white)
        self.pixmap_img = init_pixmap_img
        self.image_arr = np.zeros((800,800,3),dtype=np.uint8)
        self.ShowImgLabel.setPixmap(init_pixmap_img)
        self.OpenImgButton.clicked.connect(self.openImg)
        self.SegButton.clicked.connect(self.SegImg)
        self.MouseLeftPress = False
        self.MouseRightPress = False
        self.setWindowTitle('Tissue Annotation')

        sv_tsv = "H:\\my_anno.tsv"
        self.EditSaveTSV.setText(sv_tsv)
        if not os.path.exists(sv_tsv):
            try:
                fp = open(sv_tsv, 'w')
                fp.write("class\tR\tG\tB\n")
                fp.close()
            except:
                raise Exception("Can't create file to save annotations.")
        self.setFixedSize(self.layout().sizeHint())

    def openImg(self):
        filename = self.openWSIDialog()
        print(filename)
        if os.path.exists(filename):
            self.EditImgName.setText(filename)
            # Img = Image.open(filename).convent("RGB")
            Img = Image.open(filename)

            base_size = 800
            if Img.size[0] > Img.size[1]:
                wpercent = (base_size / float(Img.size[0]))
                hsize = int((float(Img.size[1]) * float(wpercent)))
                Img = Img.resize((base_size, hsize), Image.ANTIALIAS)
            else:
                hpercent = (base_size / float(Img.size[1]))
                wsize = int((float(Img.size[0]) * float(hpercent)))
                Img = Img.resize((wsize, base_size), Image.ANTIALIAS)
            print(Img.size)
        else:
            raise Exception("unable to open the file")
        t_pixmap_img = ImageQt(Img)
        self.image_arr = np.array(Img)
        self.pixmap_img = QPixmap.fromImage(t_pixmap_img)
        self.ShowImgLabel.setFixedWidth(self.pixmap_img.width())
        self.ShowImgLabel.setFixedHeight(self.pixmap_img.height())
        self.pixmap_img.detach()     # necessary, otherwise it crashes,
        #  see: https://stackoverflow.com/questions/35204123/python-pyqt-pixmap-update-crashes?rq=1
        self.ShowImgLabel.setPixmap(self.pixmap_img)

    def SegImg(self):
        return 0

    def get_adjacent(self, np_arr, pos):
        pixel0 = np_arr[pos.y(), pos.x(), :]
        pixel1 = np_arr[pos.y()-1, pos.x()-1, :]
        pixel2 = np_arr[pos.y()+1, pos.x()+1, :]
        pixel3 = np_arr[pos.y()-1, pos.x(), :]
        pixel4 = np_arr[pos.y()+1, pos.x(), :]
        pixel5 = np_arr[pos.y(), pos.x()-1, :]
        pixel6 = np_arr[pos.y(), pos.x()+1, :]
        pixel7 = np_arr[pos.y()+1, pos.x()-1, :]
        pixel8 = np_arr[pos.y()-1, pos.x()+1, :]
        return np.concatenate([pixel0, pixel1, pixel2, pixel3, pixel4, pixel5, pixel6, pixel7, pixel8])

    def pixels_str(self, pixel_arr):
        str_wrt = ""
        for p in pixel_arr:
            str_wrt += str(p) + "\t"
        return str_wrt

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.MouseButtonPress:
            if event.button() == QtCore.Qt.LeftButton:
                self.MouseLeftPress = True
                self.MouseRightPress = False
                #print(obj.objectName(), "Left click")
            if event.button() == QtCore.Qt.RightButton:
                self.MouseLeftPress = False
                self.MouseRightPress = True
                #print(obj.objectName(), "Right click")
        if event.type() == QtCore.QEvent.MouseMove:
            pos = event.pos()
            if self.pixmap_img:
                # image = self.pixmap_img.toImage()
                # b = image.bits()
                # # sip.voidptr must know size to support python buffer interface
                # b.setsize(self.ShowImgLabel.height() * self.ShowImgLabel.width() * 3)
                # arr = np.frombuffer(b, np.uint8).reshape((self.ShowImgLabel.height(), self.ShowImgLabel.width(), 3))
                # # pixels = arr[pos.y(), pos.x(),:]
                # pixels = self.image_arr[pos.y(), pos.x(),:]
                pixels = self.get_adjacent(self.image_arr, pos)
                print(pos)
                print(pixels)
                fp = open(self.EditSaveTSV.text(), "a")
                if self.MouseLeftPress:
                    pen = QPen(Qt.blue, 2)  # Mouse move with LEFT key pressed
                    fp.write("1\t" + self.pixels_str(pixels) + "\n")
                elif self.MouseRightPress:
                    pen = QPen(Qt.red, 2)   # Mouse move with RIGHT key pressed
                    fp.write("0\t" + self.pixels_str(pixels) + "\n")
                else:
                    fp.close()
                    raise Exception("Undefined mouse key")
                fp.close()
                painter = QPainter(self)
                painter.begin(self.pixmap_img)
                painter.setPen(pen)
                painter.drawPoint(pos)
                painter.end()
                self.ShowImgLabel.setPixmap(self.pixmap_img)
        return QtCore.QObject.event(obj, event)


    def openWSIDialog(self):
        dialog = QFileDialog(self, "Select a image")
        dialog.setFileMode(QFileDialog.AnyFile)
        # dialog.setNameFilter(str("Images (*.npg/*jpg)"))
        dialog.setViewMode(QFileDialog.Detail)
        if dialog.exec_():
            fileNames = dialog.selectedFiles()
            return fileNames[0]
        else:

            return ""

if __name__ == '__main__':
    app = QApplication(sys.argv)
    Main_Window = StainPenMarking()
    Main_Window.show()
    sys.exit(app.exec_())

