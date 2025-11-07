import numpy as np
import matplotlib.pyplot as plt
import math
from PyQt5.QtWidgets import QMainWindow, QApplication, QAction, qApp, QGridLayout, QPushButton
from PyQt5.QtWidgets import *
import sys
import pyqtgraph as pg
from PyQt5.QtCore import QSize
from scipy.optimize import curve_fit
from BrownianMotionAnalyze import BrownianMotion
from PlotWidgetPerso import PlotWidgetPerso
from PyQt5.QtCore import QTimer


class WindowFit(QWidget):
    def __init__(self, brownian):
        super().__init__()
        self.setFixedSize(QSize(800, 900))

        self.style = '''
        QGroupBox#BigGroupBox {
            border: 2px solid black;
            border-radius: 5px;
            margin-top: 20px;
            font-weight: bold;
            font-size: 15px;
        }
        QGroupBox#BigGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding-left: 10px;
            padding-right: 10px;
            padding-top: 8px;
        }
        '''

        self.style2 = '''
        QGroupBox#SmallGroupBox {
            border: 1px solid black;
            border-radius: 3px;
            margin-top: 10px;
            font-size: 13px;
            font-weight: bold;
            
        }
        QGroupBox#SmallGroupBox::title {
            subcontrol-origin: margin;
            left : 6px;
            padding-left: 0px;
            padding-right: 0px;
            padding-top: 0px;
        }
        '''

        self.init_UI(brownian)

    def init_UI(self, brownian):
        self.grid = QGridLayout()
        self.setLayout(self.grid)

        # ======== Bande de fréquence
        self.box_range_freq = QGroupBox('Frequency range to fit')

        self.grid.addWidget(self.box_range_freq, 0, 0, 1, 3)
        self.boxv_range_freq = QVBoxLayout()
        self.box_range_freq.setLayout(self.boxv_range_freq)

        self.box_range_freq.setObjectName("BigGroupBox")
        self.box_range_freq.setStyleSheet(self.style)

        self.box_freq_start = QHBoxLayout()
        self.boxv_range_freq.addLayout(self.box_freq_start)
        self.box_freq_start.addWidget(QLabel('Start frequency (Hz) :'))
        self.input_freq_start = QLineEdit()
        self.box_freq_start.addWidget(self.input_freq_start)

        self.box_freq_end = QHBoxLayout()
        self.boxv_range_freq.addLayout(self.box_freq_end)
        self.box_freq_end.addWidget(QLabel('End frequency (Hz) :'))
        self.input_freq_end = QLineEdit()
        self.box_freq_end.addWidget(self.input_freq_end)

        self.input_freq_start.setText(str(0.1))
        self.input_freq_end.setText(str(1e4))

        self.freq_start = float(self.input_freq_start.text())
        self.freq_end = float(self.input_freq_end.text())

        # fit de la psd
        self.group_fit = QGroupBox("PSD fit")
        self.group_fit.setObjectName("BigGroupBox")
        self.group_fit.setStyleSheet(self.style)
        self.grid.addWidget(self.group_fit, 1, 0, 1, 2)
        self.box_fit = QGridLayout()
        self.group_fit.setLayout(self.box_fit)

        self.button_fit = QPushButton('Fit')
        self.box_fit.addWidget(self.button_fit, 0, 0, 1, 3)
        self.button_fit.clicked.connect(lambda:
                                        self.clicked_button_fit(brownian))

        self.box_fit.addWidget(QLabel("Cut-off frequency : "), 1, 0)
        self.plot_freq_coupurex = QLabel("fcx = ")
        self.box_fit.addWidget(self.plot_freq_coupurex, 2, 0)
        self.plot_freq_coupurey = QLabel("fcy = ")
        self.box_fit.addWidget(self.plot_freq_coupurey, 3, 0)

        self.box_fit.addWidget(QLabel("Viscosity : "), 1, 1)
        self.plot_viscox = QLabel("ηx = ")
        self.box_fit.addWidget(self.plot_viscox, 2, 1)
        self.plot_viscoy = QLabel("ηy = ")
        self.box_fit.addWidget(self.plot_viscoy, 3, 1)

        self.box_fit.addWidget(QLabel("Stiffness : "), 1, 2)
        self.plot_raideurx = QLabel("kx = ")
        self.box_fit.addWidget(self.plot_raideurx, 2, 2)
        self.plot_raideury = QLabel("ky = ")
        self.box_fit.addWidget(self.plot_raideury, 3, 2)

        self.box_fit.addWidget(QLabel("  "), 4, 0)

        self.boxhvisco = QHBoxLayout()
        self.box_fit.addLayout(self.boxhvisco, 5, 0, 1, 3)
        self.boxhvisco.addWidget(QLabel("To have η = "))
        self.plot_visco_th = QLineEdit("1")
        self.boxhvisco.addWidget(self.plot_visco_th)
        self.boxhvisco.addWidget(
            QLabel("mPa.s, calibration factors have to be :"))
        self.plot_Qx_expect = QLabel("Qx = ")
        self.plot_Qy_expect = QLabel("Qy = ")
        self.box_fit.addWidget(self.plot_Qx_expect, 6, 0)
        self.box_fit.addWidget(self.plot_Qy_expect, 7, 0)

        # calcul de la pente de la psd
        self.group_slope = QGroupBox("PSD slope calculation")
        self.group_slope.setObjectName("BigGroupBox")
        self.group_slope.setStyleSheet(self.style)
        self.grid.addWidget(self.group_slope, 1, 2)
        self.box_slope = QGridLayout()
        self.group_slope.setLayout(self.box_slope)

        self.button_slope = QPushButton('calculation')
        self.box_slope.addWidget(self.button_slope, 0, 0)
        self.button_slope.clicked.connect(lambda:
                                          self.clicked_button_slope(brownian))

        self.plot_slopex = QLabel("x slope = ")
        self.box_slope.addWidget(self.plot_slopex, 1, 0)
        self.plot_slopey = QLabel("y slope = ")
        self.box_slope.addWidget(self.plot_slopey, 2, 0)

        # affichage de la psd filtrée et lissée
        self.plot_psd = PlotWidgetPerso()
        self.plot_psd.setLabel('left', 'PSD (nm²/Hz)')
        self.plot_psd.setLabel('bottom', 'fréquencies (Hz)')
        self.plot_psd.setTitle("PSD")
        self.plot_psd.addLegend()
        self.grid.addWidget(self.plot_psd, 2, 0, 1, 3)

        self.setWindowTitle("PSD analysis")
        # self.show()

    def clicked_button_fit(self, brownian):
        self.display_psd(brownian)
        reduced_indexes = (brownian.frequencies >= self.freq_start) & (
            brownian.frequencies <= self.freq_end)
        reduced_freq = brownian.frequencies[reduced_indexes]
        psd_X_reduced = brownian.PSD_X2[reduced_indexes]  # psd en m²/s
        psd_Y_reduced = brownian.PSD_Y2[reduced_indexes]

        def fonction_psd(f, D, fc, alpha):
            return D/(1+(f/fc)**alpha)

        def fonction_psdlog(f, D, fc, alpha):
            return np.log10(D) - np.log10(1+(f/fc)**alpha)

        poptx, pcov = curve_fit(
            fonction_psdlog, reduced_freq, np.log10(psd_X_reduced))
        popty, pcov = curve_fit(
            fonction_psdlog, reduced_freq, np.log10(psd_Y_reduced))

        print(poptx)
        print(popty)

        pix = self.plot_psd.getPlotItem()
        pix.setLogMode(True, True)
        pix.plot(reduced_freq, fonction_psd(
            reduced_freq, poptx[0], poptx[1], poptx[2]), pen="g", name="fit x")
        pix.plot(reduced_freq, fonction_psd(
            reduced_freq, popty[0], popty[1], popty[2]), pen="k", name="fit y")

    def clicked_button_fit2(self, brownian):

        self.display_psd(brownian)
        reduced_indexes = (brownian.frequencies >= self.freq_start) & (
            brownian.frequencies <= self.freq_end)
        reduced_freq = brownian.frequencies[reduced_indexes]
        psd_X_reduced = brownian.PSD_X2[reduced_indexes]  # psd en m²/s
        psd_Y_reduced = brownian.PSD_Y2[reduced_indexes]

        def fonction_psd(f, D, fc):
            return 2*D/(1+(f/fc)**2)/(2*math.pi**2)

        def fonction_psdlog(f, D, fc):
            return np.log10(2*D/(2*math.pi**2)) - np.log10(1+(f/fc)**2)

        poptx, pcov = curve_fit(
            fonction_psdlog, reduced_freq, np.log10(psd_X_reduced))
        popty, pcov = curve_fit(
            fonction_psdlog, reduced_freq, np.log10(psd_Y_reduced))

        gammax = brownian.kb*brownian.T/(poptx[1]**2*poptx[0]*1e-18)

        kappax = 2*math.pi*gammax*poptx[1]
        viscox = gammax/(6*math.pi*brownian.bead_radius*1e-6)

        print("frequence de coupure", kappax/(2*math.pi*gammax))

        gammay = brownian.kb*brownian.T/(popty[1]**2*popty[0]*1e-18)
        kappay = 2*math.pi*gammay*popty[1]
        viscoy = gammay/(6*math.pi*brownian.bead_radius*1e-6)

        # mettre a jour affichage visco et raideur
        self.plot_viscox.setText("ηx = %.3f mPa.s" % (viscox*1e3))
        self.plot_viscoy.setText("ηy = %.3f mPa.s" % (viscoy*1e3))

        self.plot_raideurx.setText("kx = %.3f µN/m" % (kappax*1e6))
        self.plot_raideury.setText("ky = %.3f µN/m" % (kappay*1e6))

        visco_th = float(self.plot_visco_th.text())*1e-3
        print("visco", visco_th)

        Qxth = 1/np.sqrt((6*math.pi*brownian.bead_radius *
                         1e-6*visco_th)/gammax)*brownian.factx
        Qyth = 1/np.sqrt((6*math.pi*brownian.bead_radius *
                         1e-6*visco_th)/gammay)*brownian.facty

        self.plot_Qx_expect.setText("Qx = %.1f" % (Qxth))
        self.plot_Qy_expect.setText("Qy = %.1f" % (Qyth))

        print('le Qx est', Qxth)
        print('le Qy est', Qyth)

        pix = self.plot_psd.getPlotItem()
        pix.setLogMode(True, True)
        pix.plot(reduced_freq, fonction_psd(
            reduced_freq, poptx[0], poptx[1]), pen="g", name="fit x")
        pix.plot(reduced_freq, fonction_psd(
            reduced_freq, popty[0], popty[1]), pen="k", name="fit y")

        # affichage des valeurs calculées
        self.plot_freq_coupurex.setText("fcx = %.0f Hz" % poptx[1])
        self.plot_freq_coupurey.setText("fcy = %.0f Hz" % popty[1])

    def clicked_button_slope(self, browian):
        self.display_psd(browian)
        reduced_indexes = (browian.frequencies >= self.freq_start) & (
            browian.frequencies <= self.freq_end)
        reduced_freq = browian.frequencies[reduced_indexes]
        # psd en m²/s
        psd_X_reduced = browian.PSD_X2[reduced_indexes]*1e-18
        psd_Y_reduced = browian.PSD_Y2[reduced_indexes]*1e-18

        log_echf = np.log10(reduced_freq)
        log_psdx = np.log10(psd_X_reduced)
        log_psdy = np.log10(psd_Y_reduced)

        LOGechf = np.ones((len(log_echf), 2))
        LOGechf[:, 1] = log_echf
        coeffx = np.dot(np.linalg.pinv(LOGechf), log_psdx)
        coeffy = np.dot(np.linalg.pinv(LOGechf), log_psdy)

        pix = self.plot_psd.getPlotItem()
        pix.setLogMode(True, True)
        pix.plot(reduced_freq, 10**(coeffx[0]+18)
                 * reduced_freq**coeffx[1], pen="g", name="fit x")
        pix.plot(reduced_freq, 10**(coeffy[0]+18)
                 * reduced_freq**coeffy[1], pen="k", name="fit y")

        self.plot_slopex.setText("pente x = %.2f" % coeffx[1])
        self.plot_slopey.setText("pente y = %.2f" % coeffy[1])

        kthx = browian.kb*browian.T / \
            (np.std(browian.data[:, 0])**2)*1e18*1e6
        kthx = browian.kb*browian.T / \
            np.mean(
                (browian.data[:, 0]-np.mean(browian.data[:, 0]))**2)*1e18*1e6
        kthy = browian.kb*browian.T / \
            (np.std(browian.data[:, 1])**2)*1e18*1e6

        print("kx :", kthx)
        # print("kx2 :", kthx2)
        print("ky :", kthy)
        # print("ky2 :", kthy2)

    def display_psd(self, brownian):
        self.plot_psd.clear()

        self.freq_start = float(self.input_freq_start.text())
        self.freq_end = float(self.input_freq_end.text())

        self.plot_psd.setTitle("PSD lissée et filtrée")
        pix = self.plot_psd.getPlotItem()
        pix.setLogMode(True, True)
        pix.plot(brownian.frequencies, brownian.PSD_X2, pen="r", name='x')
        pix.plot(brownian.frequencies, brownian.PSD_Y2, pen="b", name='y')

        pix.plot([self.freq_start, self.freq_start], [
                 (np.min(brownian.PSD_Y2))/2, (np.max(brownian.PSD_Y2))/2], pen='k')
        pix.plot([self.freq_end, self.freq_end], [
                 (np.min(brownian.PSD_Y2))/2, (np.max(brownian.PSD_Y2))/2], pen='k')


class WindowProgress(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(QSize(500, 100))

        self.style = '''
        QGroupBox#BigGroupBox {
            border: 2px solid #76797C;
            border-radius: 5px;
            margin-top: 20px;
            font-weight: bold;
            font-size: 15px;
        }
        QGroupBox#BigGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding-left: 10px;
            padding-right: 10px;
            padding-top: 8px;
        }
        '''
        self.init_UI()

    def init_UI(self):
        self.grid = QGridLayout()
        self.setLayout(self.grid)

        self.plot_N_calcule = QLabel(str(0))
        self.plot_N_total = QLabel(str(0))

        self.grid.addWidget(self.plot_N_calcule, 0, 0)
        self.grid.addWidget(QLabel('/'), 0, 1)
        self.grid.addWidget(self.plot_N_total, 0, 2)
        self.bar = QProgressBar(self)
        self.grid.addWidget(self.bar, 0, 3)
        self._plot_estimated_time = QLabel("Estimated time remaining :")
        self.grid.addWidget(self._plot_estimated_time, 1, 0, 1, 3)

        self.value_run = True
        self.button_stop = QPushButton("Stop the analyze")
        self.grid.addWidget(self.button_stop, 2, 0, 1, 3)
        self.button_stop.clicked.connect(self.clicked_button_top)

        self.setWindowTitle("Progress bar")
        # self.show()

    def update_bar_value(self, N_calcule, N_total, estimated_time):
        self.N_calcule = N_calcule
        self.N_total = N_total
        self.estimated_time = estimated_time
        self.update_bar()

    def update_bar(self):
        self.plot_N_calcule.setText(str(self.N_calcule))
        self.plot_N_total.setText(str(self.N_total))
        value = int(self.N_calcule/self.N_total*100)
        self.bar.setValue(value)
        time_left = int(self.estimated_time*(self.N_total - self.N_calcule))
        time_min = time_left//60
        time_sec = time_left % 60

        self._plot_estimated_time.setText(
            "temps estimé : " + str(time_min)+"min"+str(time_sec)+"s")

    def clicked_button_top(self):
        print("la boucle doit stopper")
        self.value_run = False

    def init_bar(self, N_total):
        self.value_run = True
        self.plot_N_calcule.setText(str(0))
        self.plot_N_total.setText(str(N_total))


def main():
    dossier = '2023_15_11_Bille6um_Eau_Glycerol30_piegee'
    name = 'Brownian_motion_150000Hz_3.csv'
    brownian = BrownianMotion()
    brownian.from_csv_to_data(dossier + "\\" + name)
    brownian.PSD_calculation()
    brownian.prefiltred_PSD_calculation()
    brownian.smooth_PSD()
    brownian.filtred_PSD()

    mon_app = QApplication(sys.argv)
    fenetre = WindowFit(brownian)
    fenetre.show()
    fenetre.display_psd(brownian)
    sys.exit(mon_app.exec_())


def main2():

    mon_app = QApplication(sys.argv)
    fenetre = WindowProgress()
    fenetre.show()
    sys.exit(mon_app.exec_())


if __name__ == "__main__":
    main()
