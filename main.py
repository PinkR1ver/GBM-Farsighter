import sys

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import QApplication, QCheckBox, QLabel, QMainWindow, QStatusBar, QToolBar

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("GBM Farsighter")

        self.setMinimumSize(QSize(800, 450))

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


    def clickSelectImage():
        pass

    def clickSegmentationImage():
        pass

    def clickFeatureExtraction():
        pass

    def clickSubtypePrediction():
        pass

    def clickChooseModel():
        pass



app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()