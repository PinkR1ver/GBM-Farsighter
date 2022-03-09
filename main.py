import pickle
from email.mime import base
import sys
import fnmatch
import cv2
import os
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
    QVBoxLayout,
)
from PySide6 import QtGui
from net import *
from utils import *
from utils import keep_image_size_open, keep_image_size_open_gray
import torchvision
from feature_extraction import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

basePath = os.path.dirname(__file__)
original_image = 'resize_original_img.png'
segmentation_results = 'segmentation_results.png'
annotation_image = 'annotation_image.png'

if torch.cuda.is_available():
    device = 'cuda'
    print("Using cuda")
else:
    device = 'cpu'
    print("Using CPU")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        model_detect = fnmatch.filter(os.listdir(os.path.join(basePath, 'model', 'Segmentation_Model')), "*.pth")[0]
        self.modelPath = os.path.join(basePath, 'model', 'Segmentation_Model', model_detect)
        self.net = UNet().to(device)

        self.setWindowTitle("GBM Farsighter")

        self.setFixedSize(QSize(800, 450))

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

        subtype_prediction_randomForest = QAction("Random Forest", self)
        subtype_prediction_randomForest.setStatusTip("Predict GBM subtype by Random Forest")
        subtype_prediction_randomForest.triggered.connect(self.clickSubtypePredictionRandomForest)

        subtype_prediction_MLP = QAction("MLP", self)
        subtype_prediction_MLP.setStatusTip("Predict GBM subtype by multi-layer perceptron")
        subtype_prediction_MLP.triggered.connect(self.clickSubtypePredictionMLP)

        choose_model = QAction("Choose Model", self)
        choose_model.setStatusTip("Choose trained model you want")
        choose_model.triggered.connect(self.clickChooseModel)
        choose_model.setShortcut(QKeySequence("Ctrl+c"))

        about_button = QAction("About", self)
        about_button.setStatusTip("About Developer")
        about_button.triggered.connect(self.clickAbout)

        self.setStatusBar(QStatusBar(self))

        menu = self.menuBar()

        file_menu = menu.addMenu("&FILE")
        file_menu.addAction(select_image)

        tool_menu = menu.addMenu("&TOOL")
        tool_menu.addAction(segmentaion_image)
        tool_menu.addAction(feature_extraction)
        tool_submenu = tool_menu.addMenu("Subtype Prediction")
        tool_submenu.addAction(subtype_prediction_randomForest)
        tool_submenu.addAction(subtype_prediction_MLP)

        model_menu = menu.addMenu("&MODEL")
        model_menu.addAction(choose_model)

        help_menu = menu.addMenu("&HELP")
        help_menu.addAction(about_button)




    def clickSelectImage(self, s):
        file_name, _ = QFileDialog.getOpenFileName(None, "Select a image...", './', 'Image files (*.png *.jpg)')

        if file_name:
            self.file_name = file_name
            self.segmentation_label.clear()

            img = transform(keep_image_size_open_gray(self.file_name))
            torchvision.utils.save_image(img, os.path.join(basePath, 'tmp', original_image))
            img = cv2.imread(os.path.join(basePath, 'tmp', original_image), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(basePath, 'tmp', original_image), img)

            pixmap = QtGui.QPixmap(os.path.join(basePath, 'tmp', original_image))
            pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)

            self.image_label.setPixmap(pixmap)

        elif not self.image_label.pixmap():
            delattr(self, 'file_name')
        


    def clickSegmentationImage(self, s):
        if hasattr(self, 'file_name'):
            self.net.load_state_dict(torch.load(self.modelPath, map_location=torch.device(device)))
            outImage = self.net((torch.unsqueeze(transform(keep_image_size_open_gray(self.file_name)), 0)).to(device))
            outImage = (outImage>0.5).float()
            torchvision.utils.save_image(outImage, os.path.join(basePath, 'tmp', segmentation_results))
            
            img = cv2.imread(os.path.join(basePath, 'tmp', segmentation_results), cv2.IMREAD_GRAYSCALE)
            boundary = extract_boundary(np.array(img))
            cv2.imwrite(os.path.join(basePath, 'tmp', segmentation_results), img)

            img = cv2.imread(os.path.join(basePath, 'tmp', original_image), cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = np.array(img)
            img = annotation_img(img, boundary)
            cv2.imwrite(os.path.join(basePath, 'tmp', annotation_image ), img)
            

            pixmap = QtGui.QPixmap(os.path.join(basePath, 'tmp', segmentation_results))
            pixmap = pixmap.scaled(self.segmentation_label.width(), self.segmentation_label.height(), Qt.KeepAspectRatio)

            self.segmentation_label.setPixmap(pixmap)

            pixmap = QtGui.QPixmap(os.path.join(basePath, 'tmp', annotation_image))
            pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)

            self.image_label.setPixmap(pixmap)


        else:
            dlg = QMessageBox()
            dlg.setWindowTitle("Wrong!")
            dlg.setText("You don't select image")
            dlg.exec()

    def clickFeatureExtraction(self, s):
        if hasattr(self, 'file_name') and self.segmentation_label.pixmap():
            self.feature = extract_feature(os.path.join(basePath, 'tmp', original_image), os.path.join(basePath, 'tmp', segmentation_results))
            self.feature_for_predict = np.array([])
            for i, (key, val) in enumerate(self.feature.items()):
                if i > 21:
                    self.feature_for_predict = np.append(self.feature_for_predict, val)
            dlg = QDialog()
            dlg.setWindowTitle("Features")
            layout = QGridLayout()
            #print(list(self.feature.items())[5][0])
            i = 0
            j = 0
            while True:
                #print(i*5 + j)
                #print(f'{list(self.feature.items())[i*5 + j][0]}:{list(self.feature.items())[i*5 + j][1]}')
                feature_show = str(list(self.feature.items())[i*3 + j][1])
                feature_show = (feature_show[:20] + '..') if len(feature_show) > 20 else feature_show
                layout.addWidget(QLabel(f'{list(self.feature.items())[i*3 + j][0]}:{feature_show}'), i, j)
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
        

    def clickSubtypePredictionRandomForest(self, s):
        if not hasattr(self, 'features'):
            self.feature = extract_feature(os.path.join(basePath, 'tmp', original_image), os.path.join(basePath, 'tmp', segmentation_results))
            self.feature_for_predict = np.array([])
            for i, (key, val) in enumerate(self.feature.items()):
                if i > 21:
                    self.feature_for_predict = np.append(self.feature_for_predict, val)
        if hasattr(self, 'file_name') and self.segmentation_label.pixmap():
            loaded_model = pickle.load(open(os.path.join(basePath, 'model', 'Classification_Model', 'RandomForestModel.sav'), 'rb'))
            subtype_predict = loaded_model.predict(np.expand_dims(self.feature_for_predict, axis=0))
            dlg = QMessageBox()
            dlg.setWindowTitle("Subtype Prediction")
            message_show = "Subtype is: " + str(subtype_predict[0]) + "\nThis model have 90% accuracy"
            dlg.setText(message_show)
            dlg.exec()

    def clickSubtypePredictionMLP(self, s):
        if not hasattr(self, 'features'):
            self.feature = extract_feature(os.path.join(basePath, 'tmp', original_image), os.path.join(basePath, 'tmp', segmentation_results))
            self.feature_for_predict = np.array([])
            for i, (key, val) in enumerate(self.feature.items()):
                if i > 21:
                    self.feature_for_predict = np.append(self.feature_for_predict, val)
        if hasattr(self, 'file_name') and self.segmentation_label.pixmap():
            PCA_loaded = pickle.load(open(os.path.join(basePath, 'model', 'Classification_Model', 'PCA_preparation.sav'), 'rb'))
            loaded_model = pickle.load(open(os.path.join(basePath, 'model', 'Classification_Model', 'PCAMLP.sav'), 'rb'))
            self.feature_for_predict = StandardScaler().fit_transform(np.expand_dims(self.feature_for_predict, axis=0))
            self.feature_for_predict = PCA_loaded.transform(self.feature_for_predict)
            subtype_predict = loaded_model.predict(self.feature_for_predict)
            dlg = QMessageBox()
            dlg.setWindowTitle("Subtype Prediction")
            message_show = "Subtype is: " + str(subtype_predict[0]) + "\nThis model have 92% accuracy"
            dlg.setText(message_show)
            dlg.exec()

    def clickChooseModel(self, s):
        model_name, _ = QFileDialog.getOpenFileName(None, "Select a train model..", os.path.join(basePath, 'model', 'Segmentation_Model'), '(*pth)')
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

    def clickAbout(self, s):
        dlg = QDialog()
        dlg.setWindowTitle("About")
        dlg.setFixedSize(QSize(300, 300))

        About_image = QLabel()
        pixmap = QtGui.QPixmap(os.path.join(basePath, 'data', 'Ukraine.jpg'))
        pixmap = pixmap.scaled(About_image.width() / 2, About_image.height() / 2, Qt.KeepAspectRatio)
        About_image.setPixmap(pixmap)

        About_info1 = QLabel()
        About_info1.setText("We support Ukraine")

  
        About_info2 = QLabel()
        About_info2.setText("Please Refer to www.pinkr1ver.com")      

        layout = QVBoxLayout()
        layout.addWidget(About_image, alignment=Qt.AlignCenter)
        layout.addWidget(About_info1, alignment=Qt.AlignCenter)
        layout.addWidget(About_info2, alignment=Qt.AlignCenter)

        dlg.setLayout(layout)
        dlg.exec()


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()