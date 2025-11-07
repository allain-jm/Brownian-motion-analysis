import sys

from PyQt5 import QtWidgets
import pyqtgraph as pg


class PlotWidgetPerso(pg.PlotWidget):
    def __init__(self, *args, **kwargs):
        super(PlotWidgetPerso, self).__init__(*args, **kwargs)
        self.setBackground(None)
        self.setLabel('left', '')
        self.setLabel('bottom', '')
        self.setLabel('right', '')
        self.setLabel('top', '')

        self.setTitle('', color="k")

        self.getAxis('left').setTextPen('k')
        self.getAxis('left').setPen('k')
        self.getAxis('bottom').setTextPen('k')
        self.getAxis('bottom').setPen('k')
        self.getAxis('right').setTextPen('k')
        self.getAxis('right').setPen('k')
        self.getAxis('top').setTextPen('k')
        self.getAxis('top').setPen('k')
        self.getAxis('top').setStyle(showValues=False)
        self.getAxis('right').setStyle(showValues=False)

        self.showGrid(x=True, y=True, alpha=0.15)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.plot()

    def plot(self):
        self.graph = PlotWidgetPerso()
        self.graph.setLabel('bottom', 'frequence')
        self.graph.setLabel('left', 'module')
        self.graph.setTitle('quelque chose d''utile')
        self.setCentralWidget(self.graph)


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
