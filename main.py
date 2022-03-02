from email.mime import base
import sys

import cv2
import os
from cv2 import transform
from matplotlib import widgets
from numpy import squeeze
import torch
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QWidget, 
    QApplication, 
    QLabel, 
    QMainWindow, 
    QStatusBar, 
    QFileDialog, 
    QHBoxLayout, 
    QMessageBox,
    QDialog,
    QGridLayout,
)
from PySide6 import QtGui
from net import *
from utils import *
from utils import keep_image_size_open, keep_image_size_open_gray
import torchvision
from feature_extraction import *

basePath = os.path.dirname(__file__)

if torch.cuda.is_available():
    device = 'cuda'
    print("Using cuda")
else:
    device = 'cpu'
    print("Using CPU")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.modelPath = os.path.join(basePath, 'model', 'FLAIR_unet.pth')
        self.net = UNet().to(device)

        self.setWindowTitle("GBM Farsighter")

        self.setMinimumSize(QSize(800, 450))

        self.image_label = QLabel()
        self.segmentation_label = QLabel()
        layout = QHBoxLayout()
        
        layout.addWidget(self.image_label)
        layout.addWidget(self.segmentation_label)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
          

        select_image = QAction("Select Image", self)
        select_image.setStatusTip("Select GBM MRI image to analysis")
        select_image.triggered.connect(self.clickSelectImage)
        select_image.setShortcut(QKeySequence("Ctrl+i"))

        segmentaion_image = QAction("Segmentaion", self)
        segmentaion_image.setStatusTip("Segmentate GBM")
        segmentaion_image.triggered.connect(self.clickSegmentationImage)
        segmentaion_image.setShortcut(QKeySequence("Ctrl+s"))

        feature_extraction = QAction("Feature Extraction", self)
        feature_extraction.setStatusTip("Extract features from ROI")
        feature_extraction.triggered.connect(self.clickFeatureExtraction)
        feature_extraction.setShortcut(QKeySequence("Ctrl+e"))

        subtype_prediction = QAction("Subtype Prediction", self)
        subtype_prediction.setStatusTip("Predict GBM subtype")
        subtype_prediction.triggered.connect(self.clickSubtypePrediction)
        subtype_prediction.setShortcut(QKeySequence("Ctrl+p"))

        choose_model = QAction("Choose Model", self)
        choose_model.setStatusTip("Choose trained model you want")
        choose_model.triggered.connect(self.clickChooseModel)
        choose_model.setShortcut(QKeySequence("Ctrl+c"))

        self.setStatusBar(QStatusBar(self))

        menu = self.menuBar()

        file_menu = menu.addMenu("&FILE")
        file_menu.addAction(select_image)

        tool_menu = menu.addMenu("&TOOL")
        tool_menu.addAction(segmentaion_image)
        tool_menu.addAction(feature_extraction)
        tool_menu.addAction(subtype_prediction)

        model_menu = menu.addMenu("&MODEL")
        model_menu.addAction(choose_model)


    def clickSelectImage(self, s):
        file_name, _ = QFileDialog.getOpenFileName(None, "Select a image...", './', 'Image files (*.png *.jpg)')

        if file_name:
            self.file_name = file_name
            self.segmentation_label.clear()

            img = transform(keep_image_size_open_gray(self.file_name))
            torchvision.utils.save_image(img, os.path.join(basePath, 'tmp', 'resize_original_img.png'))
            img = cv2.imread(os.path.join(basePath, 'tmp', 'resize_original_img.png'), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(basePath, 'tmp', 'resize_original_img.png'), img)

            pixmap = QtGui.QPixmap(os.path.join(basePath, 'tmp', 'resize_original_img.png'))
            pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)

            self.image_label.setPixmap(pixmap)

        elif not self.image_label.pixmap():
            delattr(self, 'file_name')
        


    def clickSegmentationImage(self, s):
        if hasattr(self, 'file_name'):
            self.net.load_state_dict(torch.load(self.modelPath, map_location=torch.device(device)))
            outImage = self.net((torch.unsqueeze(transform(keep_image_size_open_gray(self.file_name)), 0)).to(device))
            torchvision.utils.save_image(outImage, os.path.join(basePath, 'tmp', 'segmentation_results.png'))
            img = cv2.imread(os.path.join(basePath, 'tmp', 'segmentation_results.png'), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(basePath, 'tmp', 'segmentation_results.png'), img)

            pixmap = QtGui.QPixmap(os.path.join(basePath, 'tmp', 'segmentation_results.png'))
            pixmap = pixmap.scaled(self.segmentation_label.width(), self.segmentation_label.height(), Qt.KeepAspectRatio)

            self.segmentation_label.setPixmap(pixmap)
        else:
            dlg = QMessageBox()
            dlg.setWindowTitle("Wrong!")
            dlg.setText("You don't select image")
            dlg.exec()

    def clickFeatureExtraction(self, s):
        if hasattr(self, 'file_name') and self.segmentation_label.pixmap():
            self.feature = extract_feature(os.path.join(basePath, 'tmp', 'resize_original_img.png'), os.path.join(basePath, 'tmp', 'segmentation_results.png'))
            dlg = QDialog()
            dlg.setWindowTitle("Features")
            layout = QGridLayout()
            #print(list(self.feature.items())[5][0])
            i = 3
            j = 0
            while True:
                #print(i*5 + j)
                #print(f'{list(self.feature.items())[i*5 + j][0]}:{list(self.feature.items())[i*5 + j][1]}')
                layout.addWidget(QLabel(f'{list(self.feature.items())[i*3 + j][0]}:{list(self.feature.items())[i*3 + j][1]}'), i, j)
                if (i * 3 + j) >= len(self.feature) - 1:
                    break
                if j == 2:
                    j = 0
                    i += 1
                else:
                    j += 1 
            dlg.setLayout(layout)
            dlg.exec()

        elif not hasattr(self, 'file_name'):
            dlg = QMessageBox()
            dlg.setWindowTitle("Wrong!")
            dlg.setText("You don't select image")
            dlg.exec()
        
        elif not self.segmentation_label.pixmap():
            dlg = QMessageBox()
            dlg.setWindowTitle("Wrong!")
            dlg.setText("You don't annotation your image, using SEGMENTAION IMAGE")
            dlg.exec()
        

    def clickSubtypePrediction(self, s):
        print("Predict Subtyes")

    def clickChooseModel(self, s):
        model_name, _ = QFileDialog.getOpenFileName(None, "Select a train model..", os.path.join(basePath, 'model'), '(*pth)')
        if model_name:
            self.modelPath = model_name
            dlg = QMessageBox()
            dlg.setWindowTitle("Success")
            if os.name == 'nt':
                model_name = model_name.split("\\")[-1]
                dlg.setText(f'You change model into {model_name}')
            else:
                dlg.setText(f'You change model into {model_name.split("/")[-1]}')
            dlg.exec()
        
        else:
            dlg = QMessageBox()
            dlg.setWindowTitle("No Change")
            if os.name == 'nt':
                model_name = self.modelPath.split("\\")[-1]
                dlg.setText(f'You don\'t change model, you model is still {model_name}')
            else:
                dlg.setText(f'You don\'t change model, you model is still {self.modelPath.split("/")[-1]}')
            dlg.exec()


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()