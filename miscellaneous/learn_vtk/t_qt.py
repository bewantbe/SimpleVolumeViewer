import sys
from PyQt4 import QtGui, QtCore
import numpy as np

class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)

        self.pb = QtGui.QPushButton(str(np.pi), self)

app = QtGui.QApplication(sys.argv)
main = MainWindow()
main.show()
sys.exit(app.exec_())
