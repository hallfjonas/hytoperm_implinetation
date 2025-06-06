from PyQt5.QtCore import Qt, QRect, QPoint,QTimer,QSize
from PyQt5.QtGui import QPainter, QImage, QColor, QPixmap, QPen, QFont
from PyQt5.QtWidgets import  QApplication, QLabel,QMainWindow,QPushButton
import numpy as np
import math
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from scipy import ndimage
from hytoperm import *
from pprint import pprint
from LIMO_LQR import LQR
import pickle
import json

class SimWindow(QMainWindow):
    def __init__( self, parent = None,new_env=True, rbbt_viz = False) :
        super(SimWindow, self).__init__(parent)
        self.setWindowTitle("HYTOPERM Visualization")
        self.setMouseTracking(True)
        self.setGeometry(0,0,3000,1500)
        


    
if __name__ == "__main__":
