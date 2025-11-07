import numpy as np

from scipy import signal
from scipy.stats import linregress
import matplotlib.pyplot as plt
import math
from joblib import Parallel, delayed
import time
import csv
from scipy.optimize import curve_fit

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


cov = np.random.uniform(0., 10., (2, 2))
points = np.random.multivariate_normal(mean=(0, 0), cov=cov, size=1000)


class BrownianMotion:
    def __init__(self):
        self.file_name = ''  # nom du fichier
        self.data = np.array([])  # data du bruit brownien en µm
        self.fe = 65536  # frequence d'échantillonage
        self.Te = 1/self.fe  # période d'échantillon
        self.Fc = 1e4  # fréquence de coupure de la 4-quadrants
        self.bead_radius = 3  # rayon de la bille
        self.smoothing = 0.2 #valeur de base 0.2

        # facteurs de calibration
        self.factx = 1/1000
        self.facty = 1/1000

        # Constantes physiques
        self.kb = 1.38064852e-23  # constante de Boltzmann
        self.T = 293.15  # température

        self.frequencies = np.array([])  # fréquences
        self.pulsations = np.array([])  # pulsations

        # tension de la quatre quadrants
        self.voltage_X = np.array([])
        self.voltage_Y = np.array([])
        self.voltage_SUM = np.array([])

        # PSD brute
        self.PSD_X = np.array([])
        self.PSD_Y = np.array([])

        # PSD pré-filtré
        self.PSD_X_prefiltred = np.array([])
        self.PSD_Y_prefiltred = np.array([])

        # PSD lissée
        self.PSD_X_smoothed = np.array([])
        self.PSD_Y_smoothed = np.array([])

        # PSD lissée et filtrée
        self.PSD_X2 = np.array([])
        self.PSD_Y2 = np.array([])

        # fit des HF pour voir ce qui va dépasser
        # Plage de fit à HF
        self.FreqBF = 100  # freq limite BF du filtrage
        self.FreqHF = 800  # freq limite HF du filtrage
        self.FreqBF_powerlaw = 500  # freq basse  pour le calcul de la loi de puissance
        self.FreqHF_powerlaw = 2000  # Freq haute pour le calcul de la loi de puissance

        self.alpha_prime_x = np.array([])
        self.alpha_prime_y = np.array([])
        self.alpha_second_x = np.array([])
        self.alpha_second_y = np.array([])

        self.G_prime_x = np.array([])
        self.G_prime_y = np.array([])
        self.G_second_x = np.array([])
        self.G_second_y = np.array([])

        self.G_prime_m = np.array([])
        self.G_second_m = np.array([])

        # fit G' et G''
        # low pulsations fit boundaries (rad.s-1)
        self.low_puls = 30
        self.high_puls = 3000
        self.low_puls_min = 10
        self.low_puls_max = 100
        # high pulsations fit boundaries (rad.s-1)
        self.high_puls_min = 1e3
        self.high_puls_max = 1e4

        # Modules d'intérêt
        self.Gp_low_puls_final = 0
        self.Gp_high_puls_final = 0
        self.Gs_low_puls_final = 0
        self.Gs_high_puls_final = 0

    def from_csv_to_data(self, file_name):
        csvfichier = []
        self.file_name = file_name
        with open(file_name, newline="") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                csvfichier.append(row)

        self.fe = float(csvfichier[0][1])
        self.Te = 1 / self.fe
        self.time = float(csvfichier[1][1])

        self.data = []
        self.voltage_X = np.array(csvfichier[3][1:]).astype(float)
        self.voltage_Y = np.array(csvfichier[5][1:]).astype(float)
        self.voltage_SUM = np.array(csvfichier[7][1:]).astype(float)

        tabx = self.voltage_X/self.voltage_SUM
        taby = self.voltage_Y/self.voltage_SUM

        self.factx = float(csvfichier[9][1])
        self.facty = float(csvfichier[11][1])

        tabx = tabx * self.factx
        taby = taby * self.facty

        # en nm
        self.data = [tabx, taby]

        self.data = np.array(self.data)
        self.data = self.data.T

        mean_pos, width1, height1, theta1 = calcul_ellipse(self.data, 3)
        self.theta = theta1



    def rotate_data(self):

        rot_matrix = np.array([[np.cos(np.deg2rad(-self.theta)), -np.sin(np.deg2rad(-self.theta))],
                               [np.sin(np.deg2rad(-self.theta)), np.cos(np.deg2rad(-self.theta))]])
        self.data = (rot_matrix @ self.data.T).T


    def spirales_calculation(self):
        csvfichier = []
        with open(self.file_name, newline="") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                csvfichier.append(row)

        self.spirale_piezo_x = (csvfichier[13][1:])
        self.spirale_piezo_x[0] = self.spirale_piezo_x[0].replace('"[', "")
        self.spirale_piezo_x[-1] = self.spirale_piezo_x[-1].replace(']"', "")
        self.spirale_piezo_x = np.array(
            self.spirale_piezo_x).astype(float)

        self.spirale_piezo_y = (csvfichier[15][1:])
        self.spirale_piezo_y[0] = self.spirale_piezo_y[0].replace('"[', "")
        self.spirale_piezo_y[-1] = self.spirale_piezo_y[-1].replace(']"', "")
        self.spirale_piezo_y = np.array(
            self.spirale_piezo_y).astype(float)

        self.spirale_x = (csvfichier[19][1:-3])
        self.spirale_x[0] = self.spirale_x[0].replace('"[', "")
        self.spirale_x[-1] = self.spirale_x[-1].replace(']"', "")
        self.spirale_x = np.array(
            self.spirale_x).astype(float)

        self.spirale_y = (csvfichier[17][1:-3])
        self.spirale_y[0] = self.spirale_y[0].replace('"[', "")
        self.spirale_y[-1] = self.spirale_y[-1].replace(']"', "")
        self.spirale_y = np.array(
            self.spirale_y).astype(float)

        self.spirale_x = -self.spirale_x
        self.spirale_y = self.spirale_y

        # step of the spiral
        s = abs(self.spirale_piezo_y[1] - self.spirale_piezo_y[0])
        print("le step is ", s)

        self.spi_positions = np.array([-2 * s, -s, 0, s, 2 * s])

        index_X_spi = np.array([[i for i, value in enumerate(self.spirale_piezo_x) if value == pos] for pos in self.spi_positions],
                               dtype=int)
        index_Y_spi = np.array([[i for i, value in enumerate(self.spirale_piezo_y) if value == pos] for pos in self.spi_positions],
                               dtype=int)

        # Calculate mean and error for QPD positions
        self.mean_X_QPD_position = np.array(
            [np.mean(self.spirale_y[index]) - np.mean(self.spirale_y[index_Y_spi[2]]) for index in index_Y_spi])
        self.error_X_QPD_position = np.array(
            [np.std(self.spirale_y[index]) for index in index_Y_spi])

        self.mean_Y_QPD_position = np.array(
            [np.mean(self.spirale_x[index]) - np.mean(self.spirale_x[index_X_spi[2]]) for index in index_X_spi])
        self.error_Y_QPD_position = np.array(
            [np.std(self.spirale_x[index]) for index in index_X_spi])

        # Perform linear regression for calibration
        slope_x, intercept_x, _, _, _ = linregress(
            self.spi_positions, self.mean_X_QPD_position)
        slope_y, intercept_y, _, _, _ = linregress(
            self.spi_positions, self.mean_Y_QPD_position)

        self.fit_line_x = slope_x * self.spi_positions + intercept_x
        self.fit_line_y = slope_y * self.spi_positions + intercept_y

        print("calibr x ", 1/slope_x)
        print("calibr y ", 1/slope_y)

        # fig, ax = plt.subplots()
        # ax.plot(self.spi_positions, self.fit_line_x, 'b')
        # bar1 = ax.errorbar(self.spi_positions, self.mean_X_QPD_position,
        #                    yerr=self.error_X_QPD_position, marker="o", color="b",
        #                    linestyle="none", label="X QPD", capsize=2)
        # bar2 = ax.errorbar(self.spi_positions, self.mean_Y_QPD_position,
        #                    yerr=self.error_Y_QPD_position, marker="o", color="r",
        #                    linestyle="none", label="Y QPD", capsize=2)
        # ax.plot(self.spi_positions, self.fit_line_y, 'r')
        # ax.grid()
        # ax.legend()

        self.spirale_x = self.spirale_x*self.facty
        self.spirale_y = self.spirale_y*self.factx

    def smooth(self, freq, psd, smoothing):
        logmax = (np.log10(freq[-1])*2)
        if logmax % 1 != 0:
            logmax = int(logmax)
        else:
            logmax = int(logmax)-1

        local_indexes = (freq <= 1)
        ylisse = psd[local_indexes]

        for jj in range(logmax+1):
            local_indexes = (freq > 10**(jj/2)) & (freq <= 10**((jj+1)/2))
            local_freq = freq[local_indexes]
            position_local = np.where(local_indexes == True)
            N = len(local_freq)  # longueur des éléments à lisser
            Nb = smoothing*N
            Nb = int(Nb)
            if Nb % 2 == 0:
                Nb = Nb+1

            if jj == logmax:
                ylocal = psd[position_local[0][0]-Nb //
                             2: position_local[0][-1]]
                ybis = np.convolve(ylocal, np.ones(Nb)/Nb, 'same')
                ybis = ybis[Nb//2-1:]

            else:
                ylocal = psd[position_local[0][0]-Nb //
                             2: position_local[0][-1] + Nb//2+1]

                ybis = np.convolve(ylocal, np.ones(Nb)/Nb, 'valid')

            ylisse = np.append(ylisse, ybis)

        return ylisse

    def PSD_calculation(self):

        xtab = self.data[:, 0]
        ytab = self.data[:, 1]

        # Calcul de la PSD
        (self.frequencies, self.PSD_X) = signal.periodogram(
            xtab, self.fe, nfft=len(xtab))
        (self.frequencies, self.PSD_Y) = signal.periodogram(
            ytab, self.fe, nfft=len(xtab))

        self.frequencies = self.frequencies[1:]
        self.PSD_X = self.PSD_X[1:]
        self.PSD_Y = self.PSD_Y[1:]

    def prefiltred_PSD_calculation(self):
        self.PSD_X_prefiltred = self.PSD_X.copy()
        self.PSD_Y_prefiltred = self.PSD_Y.copy()

        # on commence par enlever le bruit à 50 Hz (logique)
        index50Hz = np.where((self.frequencies > 49.8) &
                             (self.frequencies < 50.2))

        # pour la valeur a 50Hz on fait la moyenne des positions adjacentes
        self.PSD_X_prefiltred[index50Hz] = (
            self.PSD_X[index50Hz[0]-1]+self.PSD_X[index50Hz[-1]+1])/2
        self.PSD_Y_prefiltred[index50Hz] = (
            self.PSD_Y[index50Hz[0]-1]+self.PSD_Y[index50Hz[-1]+1])/2

        IndF = (self.frequencies > self.FreqBF_powerlaw) & (
            self.frequencies < self.FreqHF_powerlaw)
        echff = self.frequencies[IndF]
        psdfx = self.PSD_X[IndF]
        psdfy = self.PSD_Y[IndF]

        # Passage des données en log
        logechff = np.log10(echff)
        logpsdx = np.log10(psdfx)
        logpsdy = np.log10(psdfy)

        # Fit par une droite
        LOGF = np.ones((len(logechff), 2))
        LOGF[:, 1] = logechff
        invLOGF = np.linalg.pinv(LOGF)
        bx = np.dot(invLOGF, logpsdx)
        by = np.dot(invLOGF, logpsdy)

        # Filtrage des points qui dépassent trop de la loi de puissance
        # extrapolation

        psdPLx = 10**(bx[0])*np.power(self.frequencies, bx[1])
        psdPLy = 10**(by[0])*np.power(self.frequencies, by[1])

        Ecartx = (self.PSD_X - psdPLx)/psdPLx
        Ecarty = (self.PSD_Y - psdPLy)/psdPLy

        # Détection des points trops hauts
        index2 = (self.frequencies >= self.FreqBF) & (
            self.frequencies <= self.FreqHF)
        Nb_point_cut = int(np.sum(index2*0.04))

        index = (self.frequencies < self.FreqBF) | (
            self.frequencies > self.FreqHF)
        Ecartx[index] = 0
        Ecarty[index] = 0
        list_highest_x = np.argsort(Ecartx)[-Nb_point_cut:]
        list_highest_y = np.argsort(Ecarty)[-Nb_point_cut:]

        index_last = list_highest_x[-Nb_point_cut]

        # Remplacement
        self.PSD_X_prefiltred[list_highest_x] = psdPLx[list_highest_x]
        self.PSD_Y_prefiltred[list_highest_y] = psdPLy[list_highest_y]

    def smooth_PSD(self):
        # on définit des variables locales pour optimiser la paralellisation
        freq = self.frequencies.copy()
        local_smoothing = self.smoothing
        local_PSD_X = self.PSD_X_prefiltred.copy()
        local_PSD_Y = self.PSD_Y_prefiltred.copy()

        # @numba.njit
        def smooth(y):
            logmax = (np.log10(freq[-1])*2)
            if logmax % 1 != 0:
                logmax = int(logmax)
            else:
                logmax = int(logmax)-1

            indexlocaux = (freq <= 1)
            ylisse = y[indexlocaux]

            for jj in range(logmax+1):
                indexlocaux = (freq > 10**(jj/2)
                               ) & (freq <= 10**((jj+1)/2))
                freqlocales = freq[indexlocaux]
                position_local = np.where(indexlocaux == True)
                N = len(freqlocales)  # longueur des éléments à lisser
                # print(np.where(indexlocaux == True))

                # nombre d'élément sur lesquels ont fait une moyenne
                Nb = local_smoothing*N
                Nb = int(Nb)
                if Nb % 2 == 0:
                    Nb = Nb+1

                if jj == logmax:
                    ylocal = y[position_local[0][0]-Nb //
                               2: position_local[0][-1]]
                    ybis = np.convolve(ylocal, np.ones(Nb)/Nb, 'same')
                    ybis = ybis[Nb//2-1:]

                else:
                    ylocal = y[position_local[0][0]-Nb //
                               2: position_local[0][-1] + Nb//2+1]

                    ybis = np.convolve(ylocal, np.ones(Nb)/Nb, 'valid')

                ylisse = np.append(ylisse, ybis)

            return ylisse

        psd_liss = Parallel(
            n_jobs=-1)(delayed(smooth)(psd) for psd in [local_PSD_X, local_PSD_Y])
        end = time.time()

        # print("parallel", psd_liss)
        self.PSD_X_smoothed = np.array(psd_liss[0])
        self.PSD_Y_smoothed = np.array(psd_liss[1])

    def filtred_PSD(self):
        indexes_fit = (self.frequencies > self.FreqBF_powerlaw) & (
            self.frequencies < self.FreqHF_powerlaw)
        echff = self.frequencies[indexes_fit]
        psdfx = self.PSD_X_smoothed[indexes_fit]
        psdfy = self.PSD_Y_smoothed[indexes_fit]

        # Passage des données en log
        logechff = np.log10(echff)
        logpsdx = np.log10(psdfx)
        logpsdy = np.log10(psdfy)

        # Fit par une droite
        LOGF = np.ones((len(logechff), 2))
        LOGF[:, 1] = logechff
        invLOGF = np.linalg.pinv(LOGF)
        bx = np.dot(invLOGF, logpsdx)
        by = np.dot(invLOGF, logpsdy)

        # Filtrage des points qui dépassent trop de la loi de puissance
        # extrapolation
        psdPLx = 10**(bx[0])*np.power(self.frequencies, bx[1])
        psdPLy = 10**(by[0])*np.power(self.frequencies, by[1])

        # comparaison dans la zone de test
        IndTest = (self.frequencies > self.FreqBF) & (
            self.frequencies < self.FreqHF)
        gap_X = (self.PSD_X_smoothed - psdPLx)/psdPLx
        gap_Y = (self.PSD_Y_smoothed - psdPLy)/psdPLy

        # Détection des points trops hauts / bas
        AmPic = 0.8  # ATTENTION : 20% de plus que l'écart type d'une distribution un peu bruité, cela doit être réglé
        IndFx = (np.abs(gap_X) > AmPic) & (
            self.frequencies > self.FreqBF) & (self.frequencies < self.FreqHF)
        IndFy = (np.abs(gap_Y) > AmPic) & (
            self.frequencies > self.FreqBF) & (self.frequencies < self.FreqHF)

        # Remplacement
        self.PSD_X2 = self.PSD_X_smoothed.copy()
        self.PSD_X2[IndFx] = psdPLx[IndFx]
        self.PSD_Y2 = self.PSD_Y_smoothed.copy()
        self.PSD_Y2[IndFy] = psdPLy[IndFy]

    def alpha_calculation(self, frequencies, PSD_X2, PSD_Y2):

        # Calcul de la partie imaginaire de alpha (alpha'')
        #  Le calcul est fait via le théorème de Fluctuation-Dissipation et de la PSD
        # - passage de l'échelle de fréquences en pulsations
        self.pulsations = 2*math.pi*frequencies

        self.alpha_second_x = (
            self.pulsations*PSD_X2)/(2*self.kb*self.T)
        self.alpha_second_y = (
            self.pulsations*PSD_Y2)/(2*self.kb*self.T)

        # Calcul de la partie réelle de alpha (alpha')
        #  Pour calculer l'intégrale, on va utiliser la formule de Maclaurin
        # (intégration sur des carrés) :
        # Nombre de pulsations dans le calcul d'intégrale
        Nfreq = len(self.pulsations)

        pulsations_stepsize = self.pulsations[1] - self.pulsations[0]

        # On prend les indices 2, 4, 6, pour les fréq. d'indice impairs et 1, 3, 5 pour les fréq. paires.
        ListInd1 = np.array(range(0, Nfreq-1, 2))
        ListInd2 = np.array(range(0, Nfreq-1, 2))
        ListInd2 = np.array([x+1 for x in ListInd2])
        # on créé des variables locales pour optimiser la parallélisation
        alpha_second_x_local = self.alpha_second_x.copy()
        alpha_second_y_local = self.alpha_second_y.copy()
        pulsations_local = self.pulsations.copy()

        def calcul_alpha_prime(jj):
            if jj % 2 == 0:
                ListInd = ListInd2
            else:
                ListInd = ListInd1

            pulsations_list = pulsations_local[ListInd]
            soustraction_squared = np.square(
                pulsations_list)-pulsations_local[jj]**2
            a = 2/np.pi*2*pulsations_stepsize * \
                np.sum(np.divide(pulsations_list *
                       alpha_second_x_local[ListInd], soustraction_squared))
            b = 2/np.pi*2*pulsations_stepsize * \
                np.sum(np.divide(pulsations_list *
                       alpha_second_y_local[ListInd], soustraction_squared))
            return a, b

        alpha_prime_xy = Parallel(
            n_jobs=-1)(delayed(calcul_alpha_prime)(jj) for jj in range(Nfreq))
        self.alpha_prime_x = [r[0] for r in alpha_prime_xy]
        self.alpha_prime_y = [r[1] for r in alpha_prime_xy]
        self.alpha_prime_x = np.array(self.alpha_prime_x)
        self.alpha_prime_y = np.array(self.alpha_prime_y)

        # plt.figure("alphas")
        # plt.loglog(self.pulsations, self.alpha_prime_x, '.r', label="alpha'x")
        # plt.loglog(self.pulsations, self.alpha_second_x,
        #            '.b', label="alpha''x")
        # plt.legend()
        # plt.show()

    def G_calculation(self):
        # Calculs des G' et G''
        self.G_prime_x = (1/(6*math.pi*self.bead_radius))*(self.alpha_prime_x/(self.alpha_prime_x **
                                                                               2 + self.alpha_second_x**2)) * 1e24  # formules bibliographie Gittes et Schnurr
        self.G_second_x = (1/(6*math.pi*self.bead_radius)) * \
            (self.alpha_second_x/(self.alpha_prime_x**2 + self.alpha_second_x**2)) * 1e24

        self.G_prime_y = (1/(6*math.pi*self.bead_radius))*(self.alpha_prime_y/(self.alpha_prime_y **
                                                                               2 + self.alpha_second_y**2)) * 1e24  # formules bibliographie Gittes et Schnurr
        self.G_second_y = (1/(6*math.pi*self.bead_radius)) * \
            (self.alpha_second_y/(self.alpha_prime_y**2 + self.alpha_second_y**2)) * 1e24
        # Les G' et G'' sont en Pa
        # car R est en µm et 1/alpha' en m^2/km^2/Raideur SI

        # on sort G' et G" moyens (moyenne de x et y)
        self.G_prime_m = (self.G_prime_x+self.G_prime_y)/2
        self.G_second_m = (self.G_second_x+self.G_second_y)/2

        # self.G_prime_m = (self.G_prime_y)
        # self.G_second_m = (self.G_second_y)

    def G_modulus_extraction(self, G_prime, G_second):
        low_puls_index_list = (self.pulsations > self.low_puls_min) & (
            self.pulsations < self.low_puls_max) & (G_prime > 0) & (G_second > 0)
        high_puls_index_list = (self.pulsations > self.high_puls_min) & (
            self.pulsations < self.high_puls_max) & (G_prime > 0) & (G_second > 0)
        self.low_pulsations = self.pulsations[low_puls_index_list]
        self.high_pulsations = self.pulsations[high_puls_index_list]
        G_prime_low_puls = G_prime[low_puls_index_list]
        G_second_low_puls = G_second[low_puls_index_list]
        G_prime_high_puls = G_prime[high_puls_index_list]
        G_second_high_puls = G_second[high_puls_index_list]

        # Passage des données en log
        log_low_puls = np.log10(self.low_pulsations)
        log_high_puls = np.log10(self.high_pulsations)
        log_Gp_low_puls = np.log10(G_prime_low_puls)
        log_Gs_low_puls = np.log10(G_second_low_puls)
        log_Gp_high_puls = np.log10(G_prime_high_puls)
        log_Gs_high_puls = np.log10(G_second_high_puls)

        # fonction de fit
        def f(x, a, b):
            return a*x+b

        # fits
        self.Gp_parameters_low, pcov_Gp_low = curve_fit(f, log_low_puls, log_Gp_low_puls)
        self.Gp_parameters_high, pcov_Gp_high = curve_fit(f, log_high_puls, log_Gp_high_puls)
        self.Gs_parameters_low, pcov_Gs_low = curve_fit(f, log_low_puls, log_Gs_low_puls)
        self.Gs_parameters_high, pcov_Gs_high = curve_fit(f, log_high_puls, log_Gs_high_puls)

        # Calculs des G' et G''
        self.Gp_low_puls_final = 10 ** self.Gp_parameters_low[1] * self.low_puls ** self.Gp_parameters_low[0]
        self.Gp_high_puls_final = 10 ** self.Gp_parameters_high[1] * self.high_puls ** self.Gp_parameters_high[0]
        self.Gs_low_puls_final = 10 ** self.Gs_parameters_low[1] * self.low_puls ** self.Gs_parameters_low[0]
        self.Gs_high_puls_final = 10 ** self.Gs_parameters_high[1] * self.high_puls ** self.Gs_parameters_high[0]


        #fonction de calcul d'incertitude
        def function_incert(f,popt, pcov):
            a = popt[0]
            b = popt[1]
            p_error = np.sqrt(np.diag(pcov))
            sigma_a = p_error[0]
            sigma_b = p_error[1]
            return (10 ** b * f ** a * np.log(10)) * np.sqrt((sigma_a * np.log10(f)) ** 2 + sigma_b ** 2)

        self.Gp_low_puls_error = function_incert(self.low_puls, self.Gp_parameters_low, pcov_Gp_low)
        self.Gp_high_puls_error = function_incert(self.high_puls, self.Gp_parameters_high, pcov_Gp_high)
        self.Gs_low_puls_error = function_incert(self.low_puls, self.Gs_parameters_low, pcov_Gs_low)
        self.Gs_high_puls_error = function_incert(self.high_puls, self.Gs_parameters_high, pcov_Gs_high)


        #======= ANCIENNE TECHNIQUE DE FIT
        # # Fit par une droite
        # # pour le fit, on rajoute une ligne de 1.
        # LOG_LOW_PULS = np.ones((len(log_low_puls), 2))
        # LOG_LOW_PULS[:, 1] = log_low_puls
        # LOG_HIGH_PULS = np.ones((len(log_high_puls), 2))
        # LOG_HIGH_PULS[:, 1] = log_high_puls
        #
        # #  résolution du problème linéaire => fit par b2.f+b1 des données
        # self.Gp_parameters_low = np.dot(
        #     np.linalg.pinv(LOG_LOW_PULS), log_Gp_low_puls)
        # self.Gp_parameters_high = np.dot(
        #     np.linalg.pinv(LOG_HIGH_PULS), log_Gp_high_puls)
        # self.Gs_parameters_low = np.dot(
        #     np.linalg.pinv(LOG_LOW_PULS), log_Gs_low_puls)
        # self.Gs_parameters_high = np.dot(
        #     np.linalg.pinv(LOG_HIGH_PULS), log_Gs_high_puls)
        #
        # # Modules d'intérêt
        # self.Gp_low_puls_final = 10**(
        #     self.Gp_parameters_low[0])*self.low_puls**(self.Gp_parameters_low[1])
        # self.Gp_high_puls_final = 10**(
        #     self.Gp_parameters_high[0])*self.high_puls**(self.Gp_parameters_high[1])
        # self.Gs_low_puls_final = 10**(
        #     self.Gs_parameters_low[0])*self.low_puls**(self.Gs_parameters_low[1])
        # self.Gs_high_puls_final = 10**(
        #     self.Gs_parameters_high[0])*self.high_puls**(self.Gs_parameters_high[1])






    def from_brownian_to_G(self, X, Y, sampling_frequency):
        PSD_X, PSD_Y, frequencies = self.psd_calculation(
            X, Y, sampling_frequency)
        PSD_X_smoothed, PSD_Y_smoothed = self.smooth_psd(
            PSD_X, PSD_Y, frequencies)
        alpha_prime_x, alpha_prime_y, alpha_second_x, alpha_second_y = self.alpha_calculation(
            PSD_X_smoothed, PSD_Y_smoothed)
        G_prime_m, G_second_m = self.G_calculation(
            alpha_prime_x, alpha_prime_y, alpha_second_x, alpha_second_y)
        # Gp_low_puls_final, Gp_high_puls_final, Gs_low_puls_final,Gs_high_puls_final = self.G_modulus_extraction(G_prime_m, G_second_m)

        # , Gp_low_puls_final, Gp_high_puls_final, Gs_low_puls_final, Gs_high_puls_final
        return G_prime_m, G_second_m

    def prefiltred_PSD_calculation2(self):
        self.PSD_X_prefiltred = self.PSD_X.copy()
        self.PSD_Y_prefiltred = self.PSD_Y.copy()

        # on commence par enlever le bruit à 50 Hz (logique)
        index50Hz = np.where((self.frequencies > 49.8) &
                             (self.frequencies < 50.2))

        # pour la valeur a 50Hz on fait la moyenne des positions adjacentes
        self.PSD_X_prefiltred[index50Hz] = (
            self.PSD_X[index50Hz[0]-1]+self.PSD_X[index50Hz[-1]+1])/2
        self.PSD_Y_prefiltred[index50Hz] = (
            self.PSD_Y[index50Hz[0]-1]+self.PSD_Y[index50Hz[-1]+1])/2

        IndF = (self.frequencies > self.FreqBF_powerlaw) & (
            self.frequencies < self.FreqHF_powerlaw)
        echff = self.frequencies[IndF]
        psdfx = self.PSD_X[IndF]
        psdfy = self.PSD_Y[IndF]

        # Passage des données en log
        logechff = np.log10(echff)
        logpsdx = np.log10(psdfx)
        logpsdy = np.log10(psdfy)

        # Fit par une droite
        LOGF = np.ones((len(logechff), 2))
        LOGF[:, 1] = logechff
        invLOGF = np.linalg.pinv(LOGF)
        bx = np.dot(invLOGF, logpsdx)
        by = np.dot(invLOGF, logpsdy)

        # Filtrage des points qui dépassent trop de la loi de puissance
        # extrapolation

        psdPLx = 10**(bx[0])*np.power(self.frequencies, bx[1])
        psdPLy = 10**(by[0])*np.power(self.frequencies, by[1])

        Ecartx = (self.PSD_X - psdPLx)/psdPLx
        Ecarty = (self.PSD_Y - psdPLy)/psdPLy

        # Détection des points trops hauts
        index2 = (self.frequencies >= self.FreqBF) & (
            self.frequencies <= self.FreqHF)
        Nb_point_cut = int(np.sum(index2*0.04))

        index = (self.frequencies < self.FreqBF) | (
            self.frequencies > self.FreqHF)
        Ecartx[index] = 0
        Ecarty[index] = 0
        list_highest_x = np.argsort(Ecartx)[-Nb_point_cut:]
        list_highest_y = np.argsort(Ecarty)[-Nb_point_cut:]

        index_last = list_highest_x[-Nb_point_cut]

        # Remplacement
        self.PSD_X_prefiltred[list_highest_x] = psdPLx[list_highest_x]
        self.PSD_Y_prefiltred[list_highest_y] = psdPLy[list_highest_y]
        return list_highest_x, list_highest_y


def main():

    dossier = ''
    name = 'Brownian_motion_150000Hz_1.csv'

    start = time.time()
    brownian = BrownianMotion()
    brownian.from_csv_to_data(name)

    plt.figure("brownian")
    plt.plot(brownian.data[:, 0], brownian.data[:, 1], '.')
    plt.axis("equal")

    print("file Qx", brownian.factx)
    print("file Qy", brownian.facty)

    brownian.spirales_calculation()

    print(brownian.spirale_piezo_x)
    print(brownian.spirale_piezo_y)

    plt.figure("spirale")
    plt.plot(brownian.spirale_piezo_x, brownian.spirale_piezo_y, '.-')
    plt.plot(brownian.spirale_x, brownian.spirale_y, '.-')
    plt.axis("equal")

    end = time.time()
    print('temps : ', end - start)
    plt.show()


if __name__ == "__main__":
    main()
