from email.mime import base
import sys

import cv2
import os
from cv2 import transform
import torch
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QWidget, 
    QApplication, 
    QLabel, 
    QMainWindow, 
    QStatusBar, 
    QToolBar, 
    QFileDialog, 
    QHBoxLayout, 
    QMessageBox,
)
from PySide6 import QtGui
from net import *
from utils import *
from utils import keep_image_size_open, keep_image_size_open_gray
import torchvision

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
        self.file_name, _ = QFileDialog.getOpenFileName(None, "Select a image...", './', 'Image files (*.png *.jpg)')

        if self.file_name:
            self.segmentation_label.clear()

            img = transform(keep_image_size_open_gray(self.file_name))
            torchvision.utils.save_image(img, os.path.join(basePath, 'tmp', 'resize_original_img.png'))

            pixmap = QtGui.QPixmap(os.path.join(basePath, 'tmp', 'resize_original_img.png'))
            pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)

            self.image_label.setPixmap(pixmap)

        else:
            delattr(self, 'file_name')
        


    def clickSegmentationImage(self, s):
        if hasattr(self, 'file_name'):
            self.net.load_state_dict(torch.load(self.modelPath))
            outImage = self.net((torch.unsqueeze(transform(keep_image_size_open_gray(self.file_name)), 0)).to(device))
            torchvision.utils.save_image(outImage, os.path.join(basePath, 'tmp', 'segmentation_results.png'))

            pixmap = QtGui.QPixmap(os.path.join(basePath, 'tmp', 'segmentation_results.png'))
            pixmap = pixmap.scaled(self.segmentation_label.width(), self.segmentation_label.height(), Qt.KeepAspectRatio)

            self.segmentation_label.setPixmap(pixmap)
        else:
            dlg = QMessageBox()
            dlg.setWindowTitle("Wrong!")
            dlg.setText("You don't select image")
            dlg.exec()

    def clickFeatureExtraction(self, s):
        print("Feature Extraction")

    def clickSubtypePrediction(self, s):
        print("Predict Subtyes")

    def clickChooseModel(self, s):
        print("Choose Model")



app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()