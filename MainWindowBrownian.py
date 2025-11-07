import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt
import math
from joblib import Parallel, delayed
import time
import csv
from PyQt5.QtWidgets import QMainWindow, QApplication, QAction, qApp, QGridLayout, QPushButton
from PyQt5.QtWidgets import *
import sys
from PyQt5.QtCore import QSize, Qt, QTimer

from scipy.optimize import curve_fit
import xlsxwriter
from BrownianMotionAnalyze import BrownianMotion
from WindowFitPSD import WindowFit
from WindowFitPSD import WindowProgress
from PlotWidgetPerso import PlotWidgetPerso
from matplotlib.patches import Ellipse
import pyqtgraph as pg


def calcul_ellipse(data, nstd=3):
    cov = np.cov(data, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    mean_pos = data.mean(axis=0)

    return mean_pos, width, height, theta
import scipy.stats as stats


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.brownian = BrownianMotion()
        self.window_fit = WindowFit(self.brownian)
        self.window_progress = WindowProgress()

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
            border: 1px solid gray;
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
        # self.setFixedSize(QSize(1800, 700))
        self.setMinimumSize(QSize(1800, 800))
        self.showMaximized()
        self.init_UI()
        self.show()

    def init_UI(self):
        self.grid = QGridLayout()
        self.setLayout(self.grid)

        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 1)
        self.grid.setColumnStretch(2, 1)

        self.init_group_files()
        self.grid.addWidget(self.group_files, 0, 1)

        self.init_group_brownian()
        self.grid.addWidget(self.group_brownian, 0, 0, 2, 1)

        self.init_group_PSD()
        self.grid.addWidget(self.group_psd, 1, 1, 1, 1)

        self.init_group_Gps()
        self.grid.addWidget(self.group_Gps, 0, 2, 2, 1)

        self.setWindowTitle("Main window")

    def init_group_files(self):
        self.group_files = QGroupBox("File to be analyzed")
        self.grid_files = QGridLayout()
        self.group_files.setLayout(self.grid_files)
        self.group_files.setObjectName("BigGroupBox")
        self.group_files.setStyleSheet(self.style)

        self.button_file = QPushButton('Selection of brownian file (csv)')
        self.grid_files.addWidget(self.button_file, 0, 0, 1, 2)
        self.button_file.clicked.connect(self.clicked_button_file)

        self.name_selected_folder = QLabel("Folder name  :\n")
        self.grid_files.addWidget(self.name_selected_folder, 2, 0)

        self.name_selected_file = QLabel("Selected file name :\n")
        self.grid_files.addWidget(self.name_selected_file, 2, 1)

        self.button_file_all = QPushButton(
            "Selection of a folder to analyze several brownian files")
        self.grid_files.addWidget(self.button_file_all, 3, 0, 1, 2)
        self.button_file_all.clicked.connect(self.clicked_button_file_all)

    def clicked_button_file(self):
        # on ferme toutes les fenêtre plt.figure
        plt.close("all")

        # on enregistre les anciens facteurs de calibration au cas ou on veuille les garder
        oldQx = self.brownian.factx
        oldQy = self.brownian.facty

        file = QFileDialog.getOpenFileName(self, 'Hey! Select a Files')
        #print(file)
        self.brownian.file_name = file[0]

        if file[0] == '':
            print("no file has been selected")

        elif file[0].split('.')[-1] == "csv":

            # on réinitialise les paramètres
            self.brownian.__init__()

            # on réapplique les valeurs pour l'extraction de données de G
            self.clicked_button_apply_extraction_data()

            self.brownian.file_name = file[0]
            self.brownian.from_csv_to_data(self.brownian.file_name)
            if self.rbutton_rotate.isChecked():
                self.brownian.rotate_data()
            self.brownian.PSD_calculation()
            self.brownian.prefiltred_PSD_calculation()
            self.brownian.smooth_PSD()
            self.brownian.filtred_PSD()

            self.display_new_file()

            if self.rbutton_calibr.isChecked():
                self.plotQx.setText(str(oldQx))
                self.plotQy.setText(str(oldQy))
                self.clicked_button_apply_calib()

        else:
            msg = QMessageBox()
            msg.setText("The selected file is not in csv format")
            msg.exec_()

    def clicked_button_file_all2(self):
        # ATTENTION : récolte les D, fc et les pentes des fits des PSDs
        # on enregistre les anciens facteurs de calibration au cas ou on veuille les garder
        oldQx = self.brownian.factx
        oldQy = self.brownian.facty

        msg = QMessageBox()
        msg.setText(
            "Select the folder containing the files to be analyzed.")
        msg.exec_()

        folder = QFileDialog.getExistingDirectory(
            self, 'Hey! Select a folder')
        #print(folder.split("/"))

        # récupérer tout les noms de fichiers dans le(s) dossier(s)
        def list_of_files_in_folder(list, path):
            files = os.listdir(path)
            # #print(files)
            list_local = []
            for file in files:
                if os.path.isfile(path+'/'+file):
                    # #print("c'est un fichier", file)
                    if file.endswith(".csv"):
                        list_local.append(path+'/'+file)
            if len(list_local) != 0:
                list.append(list_local)
            for file in files:
                if os.path.isdir(path+'/'+file):
                    # #print("c'est un dossier", file)
                    list_of_files_in_folder(list, path + '/' + file)

        list_folders = []

        list_of_files_in_folder(list_folders, folder)

        N_fichier = sum([len(listElem) for listElem in list_folders])

        # on ouvre le excel
        msg2 = QMessageBox()
        msg2.setText(
            "Choose the folder in which to save the results")
        msg2.exec_()
        dir_result = QFileDialog.getExistingDirectory(
            self, 'Hey! Select a folder')
        workbook = xlsxwriter.Workbook(
            dir_result + "\\resultats_" + folder.split("/")[-1] + ".xlsx")
        worksheet = workbook.add_worksheet()
        # Start from the first cell.
        # Rows and columns are zero indexed.
        row = 0
        column = 0

        worksheet.write(row, 0,  "fréquence max poure calcul de G :")
        worksheet.write(row, 1,  self.freq_max)
        row = row+2

        worksheet.write(row, 0,  "fichier")
        worksheet.write(row, 2,  "calibration x")
        worksheet.write(row, 3,  "calibration y")
        worksheet.write(row, 5,  "SNR de l'intensité du laser")
        worksheet.write(row, 7,  "Dx (nm²/Hz)")
        worksheet.write(row, 8,  "Dy (nm²/Hz)")
        worksheet.write(row, 10,  "fcx (Hz)")
        worksheet.write(row, 11,  "fcy (Hz)")
        worksheet.write(row, 13,  "alphax")
        worksheet.write(row, 14,  "alphay")

        N_calcule = 0  # nombre de fichiers calculés
        self.window_progress.init_bar(N_fichier)
        self.window_progress.show()  # on ouvre la fenetre de progress

        dir_result = dir_result + '\\' + "images_" + folder.split("/")[-1]
        if not os.path.exists(dir_result):
            os.makedirs(dir_result)

        for list_files in list_folders:
            row = row + 1
            newfolder = folder.replace('\\', '/')
            name_folder = "/".join(list(list_files[0].split('/')[0:-1]))
            name_folder = name_folder.replace(newfolder, '')
            name_folder = name_folder.replace('/', '_')
            dir_new_folder = dir_result + '\\' + name_folder
            #print("direction", dir_new_folder)

            if not os.path.exists(dir_new_folder):
                os.makedirs(dir_new_folder)

            for file_name in list_files:
                start_1_file = time.time()
                self.brownian.file_name = file_name
                self.brownian.from_csv_to_data(self.brownian.file_name)
                if self.rbutton_rotate.isChecked():
                    self.brownian.rotate_data()
                self.brownian.PSD_calculation()
                self.brownian.prefiltred_PSD_calculation()
                self.brownian.smooth_PSD()
                self.brownian.filtred_PSD()

                self.display_new_file()

                if self.rbutton_calibr.isChecked():
                    self.plotQx.setText(str(oldQx))
                    self.plotQy.setText(str(oldQy))
                    self.clicked_button_apply_calib()

                if np.std(self.brownian.voltage_SUM) != 0:
                    SNR_sum = np.mean(self.brownian.voltage_SUM) / \
                        np.std(self.brownian.voltage_SUM)
                else:
                    SNR_sum = "inf"

                # nom fichier
                newfolder = folder.replace('\\', '/')
                #print(file_name.replace(newfolder + '/', ""))
                worksheet.write(
                    row+1, 0,  file_name.replace(newfolder + '/', ""))
                # facteurs de calibration
                worksheet.write(row+1, 2,  self.brownian.factx)
                worksheet.write(row+1, 3,  self.brownian.facty)
                # SNR
                worksheet.write(row+1, 5, SNR_sum)

                reduced_indexes = (self.brownian.frequencies <= self.freq_max)
                reduced_freq = self.brownian.frequencies[reduced_indexes]
                # psd en m²/s
                psd_X_reduced = self.brownian.PSD_X2[reduced_indexes]
                psd_Y_reduced = self.brownian.PSD_Y2[reduced_indexes]

                def fonction_psd(f, D, fc, alpha):
                    return D/(1+(f/fc)**alpha)

                def fonction_psdlog(f, D, fc, alpha):
                    return np.log10(D) - np.log10(1+(f/fc)**alpha)

                poptx, pcov = curve_fit(
                    fonction_psdlog, reduced_freq, np.log10(psd_X_reduced))
                popty, pcov = curve_fit(
                    fonction_psdlog, reduced_freq, np.log10(psd_Y_reduced))

                #print(poptx)
                #print(popty)

                worksheet.write(row+1, 7, poptx[0])
                worksheet.write(row+1, 8, popty[0])
                worksheet.write(row+1, 10, poptx[1])
                worksheet.write(row+1, 11, popty[1])
                worksheet.write(row+1, 13, poptx[2])
                worksheet.write(row+1, 14, popty[2])

                row = row+1
                N_calcule = N_calcule + 1
                end_1_file = time.time()
                self.window_progress.update_bar_value(
                    N_calcule, N_fichier, end_1_file-start_1_file)
                if self.window_progress.value_run == False:
                    #print("la boucle se stop")
                    self.window_progress.close()
                    return None

                #print("temps pour un fichier", end_1_file - start_1_file)
                QApplication.processEvents()

        self.window_progress.close()

        workbook.close()

    def clicked_button_file_all(self):
        # on enregistre les anciens facteurs de calibration au cas ou on veuille les garder
        oldQx = self.brownian.factx
        oldQy = self.brownian.facty

        msg = QMessageBox()
        msg.setText(
            "Select the folder containing the files to be analyzed.")
        msg.exec_()

        folder = QFileDialog.getExistingDirectory(
            self, 'Hey! Select a folder')
        #print(folder.split("/"))

        # récupérer tout les noms de fichiers dans le(s) dossier(s)
        def list_of_files_in_folder(list, path):
            files = os.listdir(path)
            # #print(files)
            list_local = []
            for file in files:
                if os.path.isfile(path+'/'+file):
                    # #print("c'est un fichier", file)
                    if file.endswith(".csv"):
                        list_local.append(path+'/'+file)
            if len(list_local) != 0:
                list.append(list_local)
            for file in files:
                if os.path.isdir(path+'/'+file):
                    # #print("c'est un dossier", file)
                    list_of_files_in_folder(list, path + '/' + file)

        list_folders = []

        list_of_files_in_folder(list_folders, folder)

        N_fichier = sum([len(listElem) for listElem in list_folders])

        # on ouvre le excel
        msg2 = QMessageBox()
        msg2.setText(
            "Choose the folder in which to save the results")
        msg2.exec_()
        dir_result = QFileDialog.getExistingDirectory(
            self, 'Hey! Select a folder')
        workbook = xlsxwriter.Workbook(
            dir_result + "\\resultats_" + folder.split("/")[-1] + ".xlsx")
        worksheet = workbook.add_worksheet()
        # Start from the first cell.
        # Rows and columns are zero indexed.
        row = 0
        column = 0

        worksheet.write(row, 0,  "fréquence max poure calcul de G :")
        worksheet.write(row, 1,  self.freq_max)
        row = row+2

        worksheet.write(row, 0,  "fichier")
        worksheet.write(row, 2,  "calibration x")
        worksheet.write(row, 3,  "calibration y")
        worksheet.write(row, 5,  "SNR de l'intensité du laser")
        worksheet.write(row, 7,  "centre x (nm)")
        worksheet.write(row, 8,  "centre y (nm)")
        worksheet.write(row, 10,  "sigma x (nm)")
        worksheet.write(row, 11,  "sigma y (nm)")
        worksheet.write(row, 13,  "G' à " +
                        str(int(self.brownian.low_puls)) + " rad/s (Pa)")
        worksheet.write(row, 14,  "G' à " +
                        str(int(self.brownian.high_puls))+" rad/s (Pa)")
        worksheet.write(row, 16,  "G'' à " +
                        str(int(self.brownian.low_puls)) + " rad/s (Pa)")
        worksheet.write(row, 17,  "G'' à " +
                        str(int(self.brownian.high_puls)) + " rad/s (Pa)")

        worksheet.write(row, 19,  "G'x à " +
                        str(int(self.brownian.low_puls)) + " rad/s (Pa)")
        worksheet.write(row, 20,  "G'x à " +
                        str(int(self.brownian.high_puls))+" rad/s (Pa)")
        worksheet.write(row, 22,  "G''x à " +
                        str(int(self.brownian.low_puls)) + " rad/s (Pa)")
        worksheet.write(row, 23,  "G''x à " +
                        str(int(self.brownian.high_puls)) + " rad/s (Pa)")

        worksheet.write(row, 25,  "G'y à " +
                        str(int(self.brownian.low_puls)) + " rad/s (Pa)")
        worksheet.write(row, 26,  "G'y à " +
                        str(int(self.brownian.high_puls))+" rad/s (Pa)")
        worksheet.write(row, 28,  "G''y à " +
                        str(int(self.brownian.low_puls)) + " rad/s (Pa)")
        worksheet.write(row, 29,  "G''y à " +
                        str(int(self.brownian.high_puls)) + " rad/s (Pa)")

        worksheet.write(row, 31, "intensité laser sur 4quadrants(V)")
        worksheet.write(row, 33, "angle du brownien (°)")

        N_calcule = 0  # nombre de fichiers calculés
        self.window_progress.init_bar(N_fichier)
        self.window_progress.show()  # on ouvre la fenetre de progress

        # dir_result = dir_result + '\\' + "images_" + folder.split("/")[-1]
        # if not os.path.exists(dir_result):
        #     os.makedirs(dir_result)

        for list_files in list_folders:
            row = row + 1
            # newfolder = folder.replace('\\', '/')
            # name_folder = "/".join(list(list_files[0].split('/')[0:-1]))
            # name_folder = name_folder.replace(newfolder, '')
            # name_folder = name_folder.replace('/', '_')
            # dir_new_folder = dir_result + '\\' + name_folder
            #print("direction", dir_new_folder)

            # if not os.path.exists(dir_new_folder):
            #     os.makedirs(dir_new_folder)

            for file_name in list_files:
                start_1_file = time.time()
                self.brownian.file_name = file_name
                self.brownian.from_csv_to_data(self.brownian.file_name)
                if self.rbutton_rotate.isChecked():
                    self.brownian.rotate_data()
                self.brownian.PSD_calculation()
                self.brownian.prefiltred_PSD_calculation()
                self.brownian.smooth_PSD()
                self.brownian.filtred_PSD()

                self.display_new_file()

                if self.rbutton_calibr.isChecked():
                    self.plotQx.setText(str(oldQx))
                    self.plotQy.setText(str(oldQy))
                    self.clicked_button_apply_calib()

                # plot de la PSD lissée et filtrée
                plt.close("all")
                self.plot_plt_psd()
                nom_fig = "".join(list(file_name.split('/')[-1]))
                nom_fig = nom_fig.replace('.csv', '')

                # print(dir_new_folder + '\\' +
                #       "psd_lisse_filtre" + nom_fig + ".pdf")
                # 
                # plt.savefig(dir_new_folder + '\\' +
                #             "psd_lisse_filtre"+nom_fig+".pdf")


                sigmax = np.std(self.brownian.data[:, 0])
                sigmay = np.std(self.brownian.data[:, 1])

                centrex = np.mean(self.brownian.data[:, 0])
                centrey = np.mean(self.brownian.data[:, 1])

                mean_sum = np.mean(self.brownian.voltage_SUM)

                if np.std(self.brownian.voltage_SUM) != 0:
                    SNR_sum = np.mean(self.brownian.voltage_SUM) / \
                        np.std(self.brownian.voltage_SUM)

                else:
                    SNR_sum = "inf"

                # nom fichier
                newfolder = folder.replace('\\', '/')
                #print(file_name.replace(newfolder + '/', ""))
                worksheet.write(
                    row+1, 0,  file_name.replace(newfolder + '/', ""))
                # facteurs de calibration
                worksheet.write(row+1, 2,  self.brownian.factx)
                worksheet.write(row+1, 3,  self.brownian.facty)
                # SNR
                worksheet.write(row+1, 5, SNR_sum)
                # centres
                worksheet.write(row+1, 7,  centrex)
                worksheet.write(row+1, 8,  centrey)
                # sigmas
                worksheet.write(row+1, 10,  sigmax)
                worksheet.write(row+1, 11,  sigmay)
                worksheet.write(row+1, 31, mean_sum)

                worksheet.write(row + 1, 33, self.brownian.theta)



                name = ["", "x", "y"]
                self.clicked_button_G_calculation()
                for i in range(2, -1, -1):
                    #print(i)
                    startlocal = time.time()
                    if i == 0:
                        self.rbutton_Gm.click()
                    elif i == 1:
                        self.rbutton_Gx.click()
                    elif i == 2:
                        self.rbutton_Gy.click()

                    endlocal = time.time()
                    #print("temps d'un G : ", endlocal - startlocal)

                    # if i == 0:
                    #     self.plot_plt_G()
                    #     plt.savefig(dir_new_folder + '\\' +
                    #                 "G"+name[i]+nom_fig+".pdf")

                    # G'
                    worksheet.write(
                        row+1, 13 + i*6,  self.brownian.Gp_low_puls_final)
                    worksheet.write(
                        row+1, 14+i*6,  self.brownian.Gp_high_puls_final)
                    # G''
                    worksheet.write(
                        row+1, 16+i*6,  self.brownian.Gs_low_puls_final)
                    worksheet.write(
                        row+1, 17+i*6,  self.brownian.Gs_high_puls_final)

                    worksheet.write(row + 1, 35, self.brownian.Gp_low_puls_error)

                row = row+1
                N_calcule = N_calcule + 1
                end_1_file = time.time()
                self.window_progress.update_bar_value(
                    N_calcule, N_fichier, end_1_file-start_1_file)
                if self.window_progress.value_run == False:
                    #print("la boucle se stop")
                    self.window_progress.close()
                    return None

                #print("temps pour un fichier", end_1_file - start_1_file)
                QApplication.processEvents()

        self.window_progress.close()

        workbook.close()

    def connect_rbutton_rotate(self):
        if self.rbutton_rotate.isChecked():
            self.brownian.rotate_data()
        else :
            self.brownian.from_csv_to_data(self.brownian.file_name)

        self.brownian.PSD_calculation()
        self.brownian.prefiltred_PSD_calculation()
        self.brownian.smooth_PSD()
        self.brownian.filtred_PSD()

        self.display_new_file()

        if self.rbutton_calibr.isChecked():
            self.plotQx.setText(str(oldQx))
            self.plotQy.setText(str(oldQy))
            self.clicked_button_apply_calib()

    def init_group_brownian(self):
        self.group_brownian = QGroupBox("Brownian")
        self.grid_brownian = QGridLayout()
        self.group_brownian.setLayout(self.grid_brownian)
        self.group_brownian.setObjectName("BigGroupBox")
        self.group_brownian.setStyleSheet(self.style)

        self.grid_brownian.setColumnStretch(0, 1)
        self.grid_brownian.setColumnStretch(1, 1)
        self.grid_brownian.setColumnStretch(2, 1)

        self.grid_brownian.setRowStretch(0, 3)
        self.grid_brownian.setRowStretch(1, 1)
        self.grid_brownian.setRowStretch(2, 1)
        self.grid_brownian.setRowStretch(3, 1)
        self.grid_brownian.setRowStretch(4, 1)
        self.grid_brownian.setRowStretch(5, 1)
        self.grid_brownian.setRowStretch(6, 1)

        # affichage des écarts types du browien
        self.group_sigma = QGroupBox("Standard deviations")
        self.group_sigma.setObjectName("SmallGroupBox")
        self.group_sigma.setStyleSheet(self.style2)
        self.grid_brownian.addWidget(self.group_sigma, 4, 2)
        self.box_sigma = QVBoxLayout()
        self.group_sigma.setLayout(self.box_sigma)
        self.name_sigmax = QLabel("σx =")
        self.box_sigma.addWidget(self.name_sigmax)
        self.name_sigmay = QLabel("σy =")
        self.box_sigma.addWidget(self.name_sigmay)
        self.name_theta = QLabel("angle =")
        self.box_sigma.addWidget(self.name_theta)

        # affichage du centre du browien
        self.group_centre = QGroupBox("Center")
        self.group_centre.setObjectName("SmallGroupBox")
        self.group_centre.setStyleSheet(self.style2)
        self.grid_brownian.addWidget(self.group_centre, 3, 2)
        self.box_centre = QVBoxLayout()
        self.group_centre.setLayout(self.box_centre)
        self.name_centrex = QLabel("x0 =")
        self.box_centre.addWidget(self.name_centrex)
        self.name_centrey = QLabel("y0 =")
        self.box_centre.addWidget(self.name_centrey)

        # affichage du SNR de l'intensité du laser
        self.plot_SNR = QLabel("QPD sum SNR =")
        self.grid_brownian.addWidget(self.plot_SNR, 2, 2)

        # facteur de calibration
        self.group_calibr = QGroupBox("Calibration factors")
        self.group_calibr.setObjectName("SmallGroupBox")
        self.group_calibr.setStyleSheet(self.style2)
        self.grid_brownian.addWidget(self.group_calibr, 1, 0, 2, 2)
        self.grid_calibr = QGridLayout()
        self.group_calibr.setLayout(self.grid_calibr)

        self.rbutton_calibr = QCheckBox(
            "Keep calibration factors")
        self.grid_calibr.addWidget(self.rbutton_calibr, 0, 0, 1, 3)

        self.rbutton_rotate = QCheckBox(
            "rotate brownian")
        self.grid_brownian.addWidget(self.rbutton_rotate, 1, 2, 1, 1)
        self.rbutton_rotate.clicked.connect(self.connect_rbutton_rotate)

        self.grid_calibr.addWidget(QLabel("Qx :"), 1, 0)
        self.grid_calibr.addWidget(QLabel("Qy :"), 2, 0)
        self.plotQx = QLineEdit()
        self.plotQy = QLineEdit()
        self.grid_calibr.addWidget(self.plotQx, 1, 1)
        self.grid_calibr.addWidget(self.plotQy, 2, 1)

        self.button_apply_calib = QPushButton("Apply")
        self.grid_calibr.addWidget(self.button_apply_calib, 1, 2, 2, 1)
        self.button_apply_calib.clicked.connect(
            self.clicked_button_apply_calib)

        # plot du brownien
        self.plot_brownian = PlotWidgetPerso()
        self.plot_brownian.setTitle("Brownian")
        self.plot_brownian.setLabel('left', 'Y (nm)')
        self.plot_brownian.setLabel('bottom', 'X (nm)')
        self.plot_brownian.setAspectLocked()
        self.grid_brownian.addWidget(self.plot_brownian, 3, 0, 5, 2)

        # plot des signaux
        self.plot_signals = PlotWidgetPerso()
        self.plot_signals.setTitle("QPD signals")
        self.plot_signals.setLabel('left', 'Voltage (V)')
        self.plot_signals.setLabel('bottom', 'time (s)')
        self.plot_signals.addLegend(labelTextColor="k")
        self.grid_brownian.addWidget(self.plot_signals, 0, 0, 1, 3)

        self.button_plot_brown = QPushButton("Display :\nbrownian graph")
        self.button_plot_brown.clicked.connect(
            self.clicked_button_plot_brown)
        self.grid_brownian.addWidget(self.button_plot_brown, 5, 2)

        self.button_plot_quadrant = QPushButton(
            "Display :\nQPD signals")
        self.button_plot_quadrant.clicked.connect(
            self.clicked_button_plot_quadrant)
        self.grid_brownian.addWidget(self.button_plot_quadrant, 6, 2)

        self.button_plot_spirale = QPushButton(
            "Display : \nspiral calibration")
        self.button_plot_spirale.clicked.connect(
            self.clicked_button_plot_spirale)
        self.grid_brownian.addWidget(self.button_plot_spirale, 7, 2)

    def clicked_button_apply_calib(self):
        try:
            newfactx = float(self.plotQx.text())/self.brownian.factx
            newfacty = float(self.plotQy.text())/self.brownian.facty

            self.brownian.data[:, 0] = self.brownian.data[:, 0]*newfactx
            self.brownian.data[:, 1] = self.brownian.data[:, 1]*newfacty

            self.brownian.factx = float(self.plotQx.text())
            self.brownian.facty = float(self.plotQy.text())

            self.brownian.PSD_X = self.brownian.PSD_X*newfactx**2
            self.brownian.PSD_Y = self.brownian.PSD_Y*newfacty**2

            self.brownian.PSD_X_prefiltred = self.brownian.PSD_X_prefiltred*newfactx**2
            self.brownian.PSD_Y_prefiltred = self.brownian.PSD_Y_prefiltred*newfacty**2

            self.brownian.PSD_X_smoothed = self.brownian.PSD_X_smoothed*newfactx**2
            self.brownian.PSD_Y_smoothed = self.brownian.PSD_Y_smoothed*newfacty**2

            self.brownian.PSD_X2 = self.brownian.PSD_X2*newfactx**2
            self.brownian.PSD_Y2 = self.brownian.PSD_Y2*newfacty**2

            self.display_new_file()
        except:
            msg = QMessageBox()
            msg.setText("Error :no data to analyze")
            msg.exec_()

    def display_new_file(self):
        # affiche le nom du dossier
        self.name_selected_folder.setText(
            "Folder name :\n"+self.brownian.file_name.split('/')[-2])

        # affiche le nom du fichier
        self.name_selected_file.setText(
            "Selected file name :\n"+self.brownian.file_name.split('/')[-1])

        # contient les PSDs a afficher
        self.all_psd = []
        self.all_psd.append([self.brownian.PSD_X, self.brownian.PSD_Y])
        self.all_psd.append([self.brownian.PSD_X_prefiltred,
                            self.brownian.PSD_Y_prefiltred])
        self.all_psd.append(
            [self.brownian.PSD_X_smoothed, self.brownian.PSD_Y_smoothed])
        self.all_psd.append(
            [self.brownian.PSD_X2, self.brownian.PSD_Y2])

        # mise a jour de l'affichage des facteurs de calibration
        self.plotQx.setText(str(self.brownian.factx))
        self.plotQy.setText(str(self.brownian.facty))

        # affichage des signaux de la QPD
        time = np.array(list(range(len(self.brownian.voltage_X)))
                        )/self.brownian.fe
        self.plot_signals.clear()
        self.plot_signals.plot(
            time, self.brownian.voltage_X, pen="r", name="x")
        self.plot_signals.plot(
            time, self.brownian.voltage_Y, pen="b", name="y")
        self.plot_signals.plot(
            time, self.brownian.voltage_SUM, pen="k", name="sum")

        # affichage du brownien : on limite le nombre de point a afficher à environ 10 000
        datax = self.brownian.data[:, 0]
        datay = self.brownian.data[:, 1]

        Nb_affichage_max = 10000
        if self.brownian.data[:, 1].shape[0] >= Nb_affichage_max:
            index = np.array(
                range(0, len(self.brownian.data[:, 1]), len(self.brownian.data[:, 1]) // Nb_affichage_max+1))
            datax = datax[index]
            datay = datay[index]

        self.plot_brownian.clear()
        figure_brown = self.plot_brownian.plot(
            datax, datay, pen=None, symbol="o", symbolSize=5, symbolBrush="k")
        figure_brown.setAlpha(0.1, False)
        self.calcul_chi_square_test_SUM()

        # mise à jour des valeurs du centre du brownien
        self.name_centrex.setText("x0 = %.2f nm" %
                                  np.mean(self.brownian.data[:, 0]))
        self.name_centrey.setText("y0 = %.2f nm" %
                                  np.mean(self.brownian.data[:, 1]))

        # mise à jour des valeurs d'écarts type du browien
        self.name_sigmax.setText("σx = %.2f nm" %
                                 np.std(self.brownian.data[:, 0]))
        self.name_sigmay.setText("σy = %.2f nm" %
                                 np.std(self.brownian.data[:, 1]))

        self.name_theta.setText("angle = %.2f °" %
                                 self.brownian.theta)


        # mise à jour du SNR de l'intensité du laser

        SNR_sum = np.mean(self.brownian.voltage_SUM) / \
            np.std(self.brownian.voltage_SUM)

        self.plot_SNR.setText("QPD sum SNR = %.1f" % SNR_sum)
        #print(SNR_sum)

        # initie l'affichage de la Psd filtrée et lissée
        self.rbutton_psd_filtred.click()
        self.display_psd()

        # efface les données de G' et G''
        self.plot_G.clear()
        self.valueGpBF.setText("")
        self.valueGpHF.setText("")
        self.valueGsBF.setText("")
        self.valueGsHF.setText("")

    def clicked_button_plot_brown(self):
        if self.brownian.data.shape[0] != 0:
            mean_pos, width1, height1, theta1 = calcul_ellipse(
                self.brownian.data, 3)
            #print(width1, height1, theta1)

            plt.figure("brownian")
            ax = plt.gca()
            plt.plot(self.brownian.data[:, 0],
                     self.brownian.data[:, 1], '.', alpha=0.05)
            ellipse1 = Ellipse(xy=mean_pos, width=width1, height=height1, angle=theta1,
                               edgecolor='r', fc='None', lw=2, zorder=4)
            ax.add_patch(ellipse1)
            plt.plot([mean_pos[0]-width1/2*np.cos(np.deg2rad(theta1)), mean_pos[0]+width1/2*np.cos(np.deg2rad(theta1))],
                     [mean_pos[1]-width1/2*np.sin(np.deg2rad(theta1)), mean_pos[1]+width1/2*np.sin(np.deg2rad(theta1))], 'r')
            plt.plot([mean_pos[0]-height1/2*np.cos(np.deg2rad(theta1-90)), mean_pos[0]+height1/2*np.cos(np.deg2rad(theta1-90))],
                     [mean_pos[1]-height1/2*np.sin(np.deg2rad(theta1-90)), mean_pos[1] + height1/2*np.sin(np.deg2rad(theta1-90))], 'r')
            plt.xlabel("nm")
            plt.ylabel("nm")
            plt.grid()
            plt.axis("equal")
            plt.show()
        else:
            msg = QMessageBox()
            msg.setText("Error : no data to display")
            msg.exec_()

    def clicked_button_plot_quadrant(self):
        if self.brownian.voltage_X.shape[0] != 0:

            time = np.array(list(range(len(self.brownian.voltage_X)))
                            )/self.brownian.fe
            plt.figure("QPD signals")
            plt.plot(time, self.brownian.voltage_X, label="X")
            plt.plot(time, self.brownian.voltage_Y, label="Y")
            plt.plot(time, self.brownian.voltage_SUM, label="sum")
            plt.legend()
            plt.grid()
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (V)")
            plt.show()
        else:
            msg = QMessageBox()
            msg.setText("Error : no data to display")
            msg.exec_()

    def clicked_button_plot_spirale(self):
        self.brownian.spirales_calculation()

        fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True)

        ax = axs[0]
        ax.plot(self.brownian.spirale_piezo_x,
                self.brownian.spirale_piezo_y, ".-", label="piezo")
        ax.plot(self.brownian.spirale_x,
                self.brownian.spirale_y, ".-",  label="QPD")
        ax.legend()
        ax.grid()
        ax.set_xlabel("X, nm")
        ax.set_ylabel("Y, nm")

        ax = axs[1]
        ax.plot(self.brownian.spi_positions, self.brownian.fit_line_x, 'b')
        bar1 = ax.errorbar(self.brownian.spi_positions, self.brownian.mean_X_QPD_position,
                           yerr=self.brownian.error_X_QPD_position, marker="o", color="b",
                           linestyle="none", label="X QPD", capsize=2)
        bar2 = ax.errorbar(self.brownian.spi_positions, self.brownian.mean_Y_QPD_position,
                           yerr=self.brownian.error_Y_QPD_position, marker="o", color="r",
                           linestyle="none", label="Y QPD", capsize=2)
        ax.plot(self.brownian.spi_positions, self.brownian.fit_line_y, 'r')
        ax.grid()
        ax.legend()
        ax.set_xlabel("piezo (nm)")
        ax.set_ylabel("measured spiral")
        fig.set_size_inches(16, 7)
        plt.show()

    def init_group_PSD(self):
        self.group_psd = QGroupBox("PSDs")
        self.grid_psd = QGridLayout()
        self.group_psd.setLayout(self.grid_psd)
        self.group_psd.setObjectName("BigGroupBox")
        self.group_psd.setStyleSheet(self.style)

        self.names_psd = ["Raw PSD", "Raw prefiltred PSD", "Smoothed PSD",
                          "Smoothed and filtred PSD"]

        # plot radio boutons
        self.group_button_psd = QButtonGroup(self)
        self.rbutton_psd_raw = QRadioButton('Raw PSD', self)
        self.rbutton_psd_prefiltred = QRadioButton(
            'Raw prefiltred PSD', self)
        self.rbutton_psd_smoothed = QRadioButton("Smoothed PSD", self)
        self.rbutton_psd_filtred = QRadioButton(
            'Smoothed and filtred PSD', self)
        self.grid_psd.addWidget(self.rbutton_psd_raw, 0, 0)
        self.grid_psd.addWidget(self.rbutton_psd_prefiltred, 1, 0)
        self.grid_psd.addWidget(self.rbutton_psd_smoothed, 2, 0)
        self.grid_psd.addWidget(self.rbutton_psd_filtred, 3, 0)
        self.rbutton_psd_raw.toggled.connect(self.update_chosed_psd)
        self.rbutton_psd_prefiltred.toggled.connect(self.update_chosed_psd)
        self.rbutton_psd_smoothed.toggled.connect(self.update_chosed_psd)
        self.rbutton_psd_filtred.toggled.connect(self.update_chosed_psd)
        self.group_button_psd.addButton(self.rbutton_psd_raw)
        self.group_button_psd.addButton(self.rbutton_psd_prefiltred)
        self.group_button_psd.addButton(self.rbutton_psd_smoothed)
        self.group_button_psd.addButton(self.rbutton_psd_filtred)

        # plot du choix de la fréquence maximale pour le calcul
        self.box_freq_max = QGroupBox(
            "Maximum frequency for G' and G'' calculation")
        self.box_freq_max.setObjectName("SmallGroupBox")
        self.box_freq_max.setStyleSheet(self.style2)
        self.grid_psd.addWidget(self.box_freq_max, 0, 1, 3, 1)
        self.boxh_freq_max = QHBoxLayout()
        self.box_freq_max.setLayout(self.boxh_freq_max)
        self.input_freq_max = QLineEdit()
        self.boxh_freq_max.addWidget(self.input_freq_max)

        self.freq_max = 1e4
        self.input_freq_max.setText(str(self.freq_max))

        self.button_apply_freq = QPushButton('Apply')
        self.button_apply_freq.clicked.connect(self.clicked_button_apply_freq)
        self.boxh_freq_max.addWidget(self.button_apply_freq)

        # bouton pour ouvrir la fenetre de fit
        self.button_fit_psd = QPushButton('PSD fit')
        self.button_fit_psd.clicked.connect(self.clicked_button_fit_psd)
        self.grid_psd.addWidget(self.button_fit_psd, 4, 0, 1, 2)

        # plot de la PSD
        self.plot_psd = PlotWidgetPerso()
        self.plot_psd.setLabel('left', 'nm²/Hz')
        self.plot_psd.setLabel('bottom', 'Frequencies (Hz)')
        self.plot_psd.setTitle("PSD")
        self.plot_psd.addLegend(labelTextColor="k")

        self.grid_psd.addWidget(self.plot_psd, 5, 0, 1, 2)

        self.button_plot_psd = QPushButton(
            "Display :\ncurrently selected PSD")
        self.button_plot_psd.clicked.connect(
            self.clicked_button_plot_psd)
        self.grid_psd.addWidget(self.button_plot_psd, 6, 0, 1, 2)

        self.button_save_psd = QPushButton(
            "Save :\ncurrently selected PSD")
        self.button_save_psd.clicked.connect(
            self.clicked_button_save_psd)
        self.grid_psd.addWidget(self.button_save_psd, 7, 0, 1, 2)

    def update_chosed_psd(self):
        rb = self.group_button_psd.sender()
        if rb.isChecked():
            psd_text = rb.text()
            if psd_text == "Raw PSD":
                self.numero_psd = 0
            if psd_text == "Raw prefiltred PSD":
                self.numero_psd = 1
            if psd_text == "Smoothed PSD":
                self.numero_psd = 2
            if psd_text == "Smoothed and filtred PSD":
                self.numero_psd = 3
        self.display_psd()

    def display_psd(self):
        try:
            self.plot_psd.clear()
            self.plot_psd.setTitle(self.names_psd[self.numero_psd])
            pix = self.plot_psd.getPlotItem()
            pix.setLogMode(True, True)
            pix.plot(self.brownian.frequencies,
                     self.all_psd[self.numero_psd][0], pen="r", name='x')
            pix.plot(self.brownian.frequencies,
                     self.all_psd[self.numero_psd][1], pen="b", name='y')
            pix.plot([self.freq_max, self.freq_max], [
                    (np.min(self.all_psd[self.numero_psd][0])), (np.max(self.all_psd[self.numero_psd][0]))], pen="k")

        except:
            print("il n'y a pas de brownien a afficher")

    def clicked_button_apply_freq(self):
        try:
            self.freq_max = float(self.input_freq_max.text())
            msg = QMessageBox()
            msg.setText(
                "The maximum frequency has been updated")
        except ValueError:
            msg = QMessageBox()
            msg.setText(
                "The input value is not a number")
        msg.exec_()

        self.display_psd()

    def plot_plt_psd(self):
        plt.figure(self.names_psd[self.numero_psd])
        plt.loglog(self.brownian.frequencies,
                   self.all_psd[self.numero_psd][0], label='PSD x')
        plt.loglog(self.brownian.frequencies,
                   self.all_psd[self.numero_psd][1], label='PSD y')
        plt.legend()
        plt.title(self.names_psd[self.numero_psd])
        plt.grid(which="major", linewidth=1)
        # plt.grid(which="minor", linewidth=0.2)
        plt.xlabel('Fréquence (Hz)')
        plt.ylabel('PSD (nm²/Hz)')
        figure = plt.gcf()
        figure.set_size_inches(16, 9)

    def clicked_button_plot_psd(self):
        if self.brownian.frequencies.shape[0] != 0:
            plt.close(self.names_psd[self.numero_psd])

            self.plot_plt_psd()
            plt.show()
        else:
            msg = QMessageBox()
            msg.setText("Error : no data to display")
            msg.exec_()

    def clicked_button_save_psd(self):
        if self.brownian.frequencies.shape[0] != 0:
            file = QFileDialog.getExistingDirectory(
                self, 'Hey! Select a folder')

            if file == '':
                print("aucun dossier n'a été choisi")

            else:
                plt.close(self.names_psd[self.numero_psd])
                self.plot_plt_psd()
                plt.savefig(
                    file + "\\" + self.names_psd[self.numero_psd] + ".pdf")

        else:
            msg = QMessageBox()
            msg.setText("il n'y a pas de données à afficher")
            msg.exec_()

    def clicked_button_fit_psd(self):
        self.window_fit.close()
        self.window_fit.show()
        self.window_fit.display_psd(self.brownian)

    def init_group_Gps(self):
        self.group_Gps = QGroupBox("G' and G'' calculation")
        self.grid_Gps = QGridLayout()
        self.group_Gps.setLayout(self.grid_Gps)
        self.group_Gps.setObjectName("BigGroupBox")
        self.group_Gps.setStyleSheet(self.style)

        self.numero_G = 2
        self.names_G = [
            "G' and G'' (x)", "G' and G'' (y)", "G' and G'' (average)"]

        self.group_extraction_data = QGroupBox(
            "Data extraction: selected values and fit boundaries")
        self.group_extraction_data.setObjectName("SmallGroupBox")
        self.group_extraction_data.setStyleSheet(self.style2)
        self.grid_Gps.addWidget(self.group_extraction_data, 3, 0, 1, 3)
        self.grid_extraction_data = QGridLayout()
        self.group_extraction_data.setLayout(self.grid_extraction_data)

        self.grid_extraction_data.addWidget(QLabel("Value 1 :"), 1, 0)
        self.choose_low_puls = QLineEdit()
        self.choose_low_puls_min = QLineEdit()
        self.choose_low_puls_max = QLineEdit()
        self.grid_extraction_data.addWidget(self.choose_low_puls, 1, 1)
        self.grid_extraction_data.addWidget(
            QLabel("rad/s, for a fit between"), 1, 2)
        self.grid_extraction_data.addWidget(self.choose_low_puls_min, 1, 3)
        self.grid_extraction_data.addWidget(
            QLabel("rad/s and"), 1, 4)

        self.grid_extraction_data.addWidget(self.choose_low_puls_max, 1, 5)
        self.grid_extraction_data.addWidget(
            QLabel("rad/s"), 1, 6)

        self.grid_extraction_data.addWidget(QLabel("Value 2 :"), 4, 0)
        self.choose_high_puls = QLineEdit()
        self.choose_high_puls_min = QLineEdit()
        self.choose_high_puls_max = QLineEdit()
        self.grid_extraction_data.addWidget(self.choose_high_puls, 4, 1)
        self.grid_extraction_data.addWidget(
            QLabel("rad/s, for a fit between"), 4, 2)
        self.grid_extraction_data.addWidget(self.choose_high_puls_min, 4, 3)
        self.grid_extraction_data.addWidget(
            QLabel("rad/s and"), 4, 4)
        self.grid_extraction_data.addWidget(self.choose_high_puls_max, 4, 5)
        self.grid_extraction_data.addWidget(
            QLabel("rad/s"), 4, 6)

        self.button_apply_extraction_data = QPushButton("Apply")
        self.grid_extraction_data.addWidget(
            self.button_apply_extraction_data, 5, 2, 1, 2)
        self.button_apply_extraction_data.clicked.connect(
            self.clicked_button_apply_extraction_data)

        self.group_button_G = QButtonGroup(self)
        self.rbutton_Gx = QRadioButton('x', self)
        self.rbutton_Gy = QRadioButton("y", self)
        self.rbutton_Gm = QRadioButton("average of x and y", self)
        self.grid_Gps.addWidget(self.rbutton_Gx, 0, 0)
        self.grid_Gps.addWidget(self.rbutton_Gy, 0, 1)
        self.grid_Gps.addWidget(self.rbutton_Gm, 0, 2)
        self.rbutton_Gx.toggled.connect(self.update_chosed_G)
        self.rbutton_Gy.toggled.connect(self.update_chosed_G)
        self.rbutton_Gm.toggled.connect(self.update_chosed_G)
        self.group_button_G.addButton(self.rbutton_Gx)
        self.group_button_G.addButton(self.rbutton_Gy)
        self.group_button_G.addButton(self.rbutton_Gm)
        self.rbutton_Gm.click()
        self.numero_G = 2

        self.button_calcul = QPushButton("G' and G'' calculation")
        self.button_calcul.clicked.connect(self.clicked_button_G_calculation)
        self.grid_Gps.addWidget(self.button_calcul, 2, 0, 1, 3)

        self.grid_Gps.addWidget(QLabel("Bead radius"), 1, 0)
        self.choose_bead_radius = QLineEdit()
        self.grid_Gps.addWidget(self.choose_bead_radius, 1, 1)
        self.choose_bead_radius.setText(str(self.brownian.bead_radius))
        self.choose_bead_radius.textChanged.connect(self.change_radius_bead)

        self.choose_low_puls.setText(str(self.brownian.low_puls))
        self.choose_low_puls_min.setText(str(self.brownian.low_puls_min))
        self.choose_low_puls_max.setText(str(self.brownian.low_puls_max))

        self.choose_high_puls.setText(str(self.brownian.high_puls))
        self.choose_high_puls_min.setText(str(self.brownian.high_puls_min))
        self.choose_high_puls_max.setText(str(self.brownian.high_puls_max))

        # plot des valeurs
        self.textGprime = QLabel("G'")
        self.textGprime.setAlignment(Qt.AlignCenter)
        self.grid_Gps.addWidget(self.textGprime, 4, 1)

        self.textGsec = QLabel("G''")
        self.textGsec.setAlignment(Qt.AlignCenter)
        self.grid_Gps.addWidget(self.textGsec, 4, 2)

        self.plot_pulse_low = QLabel(self.choose_low_puls.text() + " rad/s")
        self.plot_pulse_low.setAlignment(Qt.AlignCenter)
        self.grid_Gps.addWidget(self.plot_pulse_low, 4, 0)

        self.plot_pulse_high = QLabel(self.choose_high_puls.text() + " rad/s")
        self.plot_pulse_high.setAlignment(Qt.AlignCenter)
        self.grid_Gps.addWidget(self.plot_pulse_high, 6, 0)

        self.valueGpBF = QLabel("")
        self.valueGpBF.setAlignment(Qt.AlignCenter)
        self.grid_Gps.addWidget(self.valueGpBF, 5, 1)

        self.valueGpHF = QLabel("")
        self.valueGpHF.setAlignment(Qt.AlignCenter)
        self.grid_Gps.addWidget(self.valueGpHF, 6, 1)

        self.valueGsBF = QLabel("")
        self.valueGsBF.setAlignment(Qt.AlignCenter)
        self.grid_Gps.addWidget(self.valueGsBF, 5, 2)

        self.valueGsHF = QLabel("")
        self.valueGsHF.setAlignment(Qt.AlignCenter)
        self.grid_Gps.addWidget(self.valueGsHF, 6, 2)

        # plot de G' et G''
        self.plot_G = PlotWidgetPerso()
        self.plot_G.setLabel('left', 'Modulus (Pa)')
        self.plot_G.setLabel('bottom', 'pulsations (rad/s)')
        self.plot_G.setTitle("G' and G''")
        self.plot_G.addLegend(labelTextColor="k")
        self.grid_Gps.addWidget(self.plot_G, 7, 0, 1, 3)

        self.button_plot_G = QPushButton("Display  :\nG' and G'' graph")
        self.button_plot_G.clicked.connect(
            self.clicked_button_plot_G)
        self.grid_Gps.addWidget(self.button_plot_G, 8, 0, 1, 3)

        self.button_save_G = QPushButton("Save :\nG' et G'' graph")
        self.button_save_G.clicked.connect(
            self.clicked_button_save_G)
        self.grid_Gps.addWidget(self.button_save_G, 9, 0, 1, 3)

    def update_chosed_G(self):
        rb = self.group_button_G.sender()
        if rb.isChecked():
            G_text = rb.text()
            if G_text == "x":
                self.numero_G = 0
            if G_text == "y":
                self.numero_G = 1
            if G_text == "average of x and y":
                self.numero_G = 2

        if self.numero_G == 0:
            self.G_prime = self.brownian.G_prime_x.copy()
            self.G_second = self.brownian.G_second_x.copy()
        if self.numero_G == 1:
            self.G_prime = self.brownian.G_prime_y.copy()
            self.G_second = self.brownian.G_second_y
        if self.numero_G == 2:
            self.G_prime = self.brownian.G_prime_m.copy()
            self.G_second = self.brownian.G_second_m.copy()

        self.button_apply_extraction_data.click()


    def change_radius_bead(self,text):
        self.brownian.bead_radius = float(text)
        print("un changement")
        print(self.brownian.bead_radius)


    def clicked_button_G_calculation(self):
        # try:
        index_max = self.brownian.frequencies < self.freq_max
        reduced_frequencies = self.brownian.frequencies[index_max]
        psd_X2 = self.brownian.PSD_X2[index_max]
        psd_Y2 = self.brownian.PSD_Y2[index_max]

        self.brownian.alpha_calculation(
            reduced_frequencies, psd_X2, psd_Y2)

        self.brownian.G_calculation()

        self.G_prime = np.array([])
        self.G_second = np.array([])

        if self.numero_G == 0:
            self.G_prime = self.brownian.G_prime_x.copy()
            self.G_second = self.brownian.G_second_x.copy()
        if self.numero_G == 1:
            self.G_prime = self.brownian.G_prime_y.copy()
            self.G_second = self.brownian.G_second_y
        if self.numero_G == 2:
            self.G_prime = self.brownian.G_prime_m.copy()
            self.G_second = self.brownian.G_second_m.copy()

        self.brownian.G_modulus_extraction(self.G_prime, self.G_second)

        self.display_G()

        # except:
        #     msg = QMessageBox()
        #     msg.setText(
        #         "Il n'y a pas de brownien à calculer")
        #     msg.exec_()

    def display_G(self):
        # affichage sur pyqtgraph

        # il faut enlever les valeurs nulles car pyqtgraph ne gère pas bien ces valeurs en échelles log
        indexprime = (self.G_prime > 0)
        indexseconde = (self.G_second > 0)
        pulsationplotprime = self.brownian.pulsations[indexprime]
        pulsationplotseconde = self.brownian.pulsations[indexseconde]
        Gprimeplot = self.G_prime[indexprime]
        Gsecondplot = self.G_second[indexseconde]

        low_puls_range = [self.brownian.low_puls_min,
                          self.brownian.low_puls_max]
        high_puls_range = [self.brownian.high_puls_min,
                           self.brownian.high_puls_max]

        self.plot_G.clear()
        self.plot_G.setTitle(self.names_G[self.numero_G])
        pix = self.plot_G.getPlotItem()
        pix.setLogMode(True, True)
        pix.plot(pulsationplotprime,
                 Gprimeplot, pen="b", name="G'")
        pix.plot(pulsationplotseconde,
                 Gsecondplot, pen="r", name="G''")
        pix.plot(low_puls_range, 10 **
                 (self.brownian.Gp_parameters_low[1])*low_puls_range**(self.brownian.Gp_parameters_low[0]), pen="k", name="fits")
        pix.plot(high_puls_range, 10 **
                 (self.brownian.Gp_parameters_high[1])*high_puls_range**(self.brownian.Gp_parameters_high[0]), pen="k")
        pix.plot(low_puls_range, 10 **
                 (self.brownian.Gs_parameters_low[1])*low_puls_range**(self.brownian.Gs_parameters_low[0]), pen='k')
        pix.plot(high_puls_range, 10 **
                 (self.brownian.Gs_parameters_high[1])*high_puls_range**(self.brownian.Gs_parameters_high[0]), pen='k')

        #afficher errorbar
        error_Gp_low_top = np.log10(self.brownian.Gp_low_puls_final + self.brownian.Gp_low_puls_error) - np.log10(self.brownian.Gp_low_puls_final)
        error_Gp_low_bottom = -np.log10(self.brownian.Gp_low_puls_final - self.brownian.Gp_low_puls_error) + np.log10(self.brownian.Gp_low_puls_final)

        error_Gp_high_top = np.log10(self.brownian.Gp_high_puls_final + self.brownian.Gp_high_puls_error) - np.log10(
            self.brownian.Gp_high_puls_final)
        error_Gp_high_bottom = -np.log10(self.brownian.Gp_high_puls_final - self.brownian.Gp_high_puls_error) + np.log10(
            self.brownian.Gp_high_puls_final)

        error_Gs_low_top = np.log10(self.brownian.Gs_low_puls_final + self.brownian.Gs_low_puls_error) - np.log10(
            self.brownian.Gs_low_puls_final)
        error_Gs_low_bottom = -np.log10(self.brownian.Gs_low_puls_final - self.brownian.Gs_low_puls_error) + np.log10(
            self.brownian.Gs_low_puls_final)

        error_Gs_high_top = np.log10(self.brownian.Gs_high_puls_final + self.brownian.Gs_high_puls_error) - np.log10(
            self.brownian.Gs_high_puls_final)
        error_Gs_high_bottom = -np.log10(
            self.brownian.Gs_high_puls_final - self.brownian.Gs_high_puls_error) + np.log10(
            self.brownian.Gs_high_puls_final)

        err_y = pg.ErrorBarItem(
            x=np.log10(self.brownian.low_puls), y=np.log10(self.brownian.Gp_low_puls_final),
            top=error_Gp_low_top, bottom=error_Gp_low_bottom,
            beam=0.05
        )
        pix.addItem(err_y)

        pix.plot([self.brownian.low_puls],
                 [self.brownian.Gp_low_puls_final], symbol="o", symbolSize=8, symbolBrush="k")
        pix.plot([self.brownian.low_puls],
                 [self.brownian.Gs_low_puls_final], symbol="o", symbolSize=8, symbolBrush="k")
        pix.plot([self.brownian.high_puls],
                 [self.brownian.Gp_high_puls_final], symbol="o", symbolSize=8, symbolBrush="k")
        pix.plot([self.brownian.high_puls],
                 [self.brownian.Gs_high_puls_final], symbol="o", symbolSize=8, symbolBrush="k")

        # mettre les valeurs de G' et G''
        self.valueGpBF.setText("%.3f ± %.3f Pa" % (self.brownian.Gp_low_puls_final, self.brownian.Gp_low_puls_error))
        self.valueGpHF.setText("%.3f ± %.3f Pa" % (self.brownian.Gp_high_puls_final, self.brownian.Gp_high_puls_error))
        self.valueGsBF.setText("%.3f ± %.3f Pa" % (self.brownian.Gs_low_puls_final, self.brownian.Gs_low_puls_error))
        self.valueGsHF.setText("%.3f ± %.3f Pa" % (self.brownian.Gs_high_puls_final, self.brownian.Gs_high_puls_error))

    def plot_plt_G(self):
        plt.figure(self.names_G[self.numero_G])
        # plot de G' et g''
        plt.loglog(self.brownian.pulsations,
                   self.G_prime, '.b', label="G' moyen")
        plt.loglog(self.brownian.pulsations, self.G_second,
                   '.r', label="G'' moyen")

        #plot des barres d'erreur
        plt.errorbar(self.brownian.low_puls, self.brownian.Gp_low_puls_final, yerr=self.brownian.Gp_low_puls_error, fmt='o',
                     capsize=5, label="données avec erreurs")
        plt.errorbar(self.brownian.high_puls, self.brownian.Gp_high_puls_final, yerr=self.brownian.Gp_high_puls_error,
                     fmt='o',
                     capsize=5)
        plt.errorbar(self.brownian.low_puls, self.brownian.Gs_low_puls_final, yerr=self.brownian.Gs_low_puls_error,
                     fmt='o',
                     capsize=5)
        plt.errorbar(self.brownian.high_puls, self.brownian.Gs_high_puls_final, yerr=self.brownian.Gs_high_puls_error,
                     fmt='o',
                     capsize=5)
        # plot des fits et des points retenus
        low_puls_range = [self.brownian.low_puls_min,
                          self.brownian.low_puls_max]
        high_puls_range = [self.brownian.high_puls_min,
                           self.brownian.high_puls_max]

        plt.loglog(low_puls_range, 10 **
                   (self.brownian.Gp_parameters_low[1])*low_puls_range**(self.brownian.Gp_parameters_low[0]), '-k', label="fits")
        plt.loglog(high_puls_range, 10 **
                   (self.brownian.Gp_parameters_high[1])*high_puls_range**(self.brownian.Gp_parameters_high[0]), '-k')
        plt.loglog(self.brownian.low_puls, self.brownian.Gp_low_puls_final, 'ok',
                   mfc='none', label="valeurs retenues")
        plt.loglog(self.brownian.high_puls,
                   self.brownian.Gp_high_puls_final, 'ok', mfc='none')

        plt.loglog(low_puls_range, 10 **
                   (self.brownian.Gs_parameters_low[1])*low_puls_range**(self.brownian.Gs_parameters_low[0]), '-k')
        plt.loglog(high_puls_range, 10 **
                   (self.brownian.Gs_parameters_high[1])*high_puls_range**(self.brownian.Gs_parameters_high[0]), '-k')
        plt.loglog(self.brownian.low_puls,
                   self.brownian.Gs_low_puls_final, 'ok', mfc='none')
        plt.loglog(self.brownian.high_puls,
                   self.brownian.Gs_high_puls_final, 'ok', mfc='none')

        plt.legend()
        plt.title(self.names_G[self.numero_G])

        plt.xlabel('Pulsation (rad/s)')
        plt.ylabel("Module de rigidité G' (Pa)")

        plt.grid(which="major", linewidth=1)
        # plt.grid(which="minor", linewidth=0.2)

        figure = plt.gcf()
        figure.set_size_inches(16, 9)

    def clicked_button_plot_G(self):
        if self.brownian.pulsations.shape[0] != 0:
            plt.close(self.names_G[self.numero_G])
            self.plot_plt_G()
            plt.show()
        else:
            msg = QMessageBox()
            msg.setText("Error : no data to display")
            msg.exec_()

    def clicked_button_save_G(self):
        plt.close(self.names_G[self.numero_G])

        if self.brownian.pulsations.shape[0] != 0:
            file = QFileDialog.getExistingDirectory(
                self, 'Hey! Select several Files')

            if file == '':
                print("aucun dossier n'a été choisi")

            else:
                plt.close(self.names_G[self.numero_G])
                self.plot_plt_G()
                plt.savefig(file + "\\" + "G_prime_seconde.pdf")

        else:
            msg = QMessageBox()
            msg.setText("Error : no data to save")
            msg.exec_()

    def clicked_button_apply_extraction_data(self):
        try:
            self.brownian.low_puls = float(self.choose_low_puls.text())
            self.brownian.low_puls_min = float(self.choose_low_puls_min.text())
            self.brownian.low_puls_max = float(self.choose_low_puls_max.text())

            self.brownian.high_puls = float(self.choose_high_puls.text())
            self.brownian.high_puls_min = float(
                self.choose_high_puls_min.text())
            self.brownian.high_puls_max = float(
                self.choose_high_puls_max.text())

            self.plot_pulse_low.setText(self.choose_low_puls.text() + " rad/s")
            self.plot_pulse_high.setText(
                self.choose_high_puls.text() + " rad/s")
        except:
            print("pas encore de fichier")
        # msg = QMessageBox()
        # msg.setText(
        #     "Les valeurs ont été mises à jour")
        # msg.exec_()

        if self.brownian.pulsations.shape[0] != 0:
            self.brownian.G_modulus_extraction(self.G_prime, self.G_second)
            self.display_G()
        else:
            print("il n'y a pas de donnée à afficher")

    def calcul_chi_square_test_SUM(self):
        sigma0 = 0.000600175999
        sigma = np.std(self.brownian.voltage_SUM)
        n = self.brownian.voltage_SUM.shape[0]
        mean = np.mean(self.brownian.voltage_SUM)
        print("mean", mean)
        print("std", sigma)
        print("n", n)

        tn = (n-1)*sigma**2/sigma0**2

        q = stats.chi2.ppf(0.95, df=(n-1))
        pvalue = 1- stats.chi2.cdf(tn, df=(n-1))

        print("tn", tn)
        print("quantile", q)
        print("pvalue", pvalue)



def main():
    mon_app = QApplication(sys.argv)
    fenetre = MainWindow()
    sys.exit(mon_app.exec_())


if __name__ == "__main__":
    main()
