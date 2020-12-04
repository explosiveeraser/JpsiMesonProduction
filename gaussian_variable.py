#Imports Modules needed later
import numpy as np
import matplotlib.pyplot as plt
from variable import Variable
from gaussian import Gaussian
from double_gaussian import Double_gaussian
from crystal_ball import Crystal_ball
from math import floor, log10

#Function that rounds x
def round_digits(x, n):
    return round(x, -int(floor(log10(abs(x)))) + n)

#Creates a class for variables characterized by a gaussian curve with exponential background or Crystal Ball function
#shape
class Gaussian_variable(Variable): #Inherits Variable Class
    #Method to initialize the class parameters to be used later for analysis
    def __init__(self, data, limits, num_bins, title, xlabel, ylabel, file_name, fig_num, units):
        self.data = data
        print(np.min(data))
        print(np.max(data))
        self.limits = limits
        self.num_bins = num_bins
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.file_name = file_name
        self.fig_num = fig_num
        self.n = []
        self.bins = []
        self.patches = []
        self.mean = float()
        self.resolution = float()
        self.FWHM =float()
        self.signal_limits = tuple()
        self.lower_sideband_limits = tuple()
        self.upper_sideband_limits = tuple()
        self.noise_fit = list()
        self.N = float()
        self.B = float()
        self.S = float()
        self.prob = float()
        self.bin_width = (np.max(self.data)-np.min(self.data))/self.num_bins
        self.parameters_g = dict()
        self.parameters_2g = dict()
        self.parameters_cb = dict()
        self.single_gaussian = Gaussian() #utilizes gaussian module imported
        self.double_gaussian = Double_gaussian() #utilizes double gaussian module imported
        self.crystal_ball = Crystal_ball() #utilizes crystal ball module imported
        self.units = units

    #method that finds the mean of a gaussian by finding the maximum bin in a gaussian histogram
    def find_mean(self):
        largest_bin = np.amax(self.n)
        i = np.where(self.n == largest_bin)[0][0]
        self.mean = self.bins[i]
        return self.mean

    #method to set the Full Width Half Maximum after visual inspection of the histogram
    def set_FWHM(self, x):
        self.FWHM = x
        return self.FWHM

    #method to obtain an estimate of the resolutions of the histogram after finding the mean and setting the FWHM
    def find_resolution(self):
        self.resolution = self.mean/self.FWHM
        return self.resolution

    #method that finds the minimum and maximum indices that satisify a limit within the self.bin parameter
    def find_limits_index(self, limits):
        i_min = 0
        i_max = 0
        for i in range(0, len(self.bins)):
            if self.bins[i] <= limits[0]:
                i_min = i
            elif self.bins[i] >= limits[0] and self.bins[i] <= limits[1]:
                i_max = i
        return (i_min, i_max)

    #method to obtain the minimum and maximum indices that characterize the signal in the self.bin parameter.
    #The signal is the part of the curve that is most characterized by a gaussian curve
    def find_signal_limits(self):
        limits = (self.mean-30., self.mean+30)
        i_limits = self.find_limits_index(limits)
        self.signal_limits = i_limits
        return i_limits

    #method to find the number of events within the signal limits
    def num_events_signal_width(self):
        total = 0
        for j in range(self.signal_limits[0], self.signal_limits[1]):
            total += self.n[j]
        self.N = float(total)
        return total

    #method that finds the minimum and maximum indices of the limits for both sidebands on either side of
    #the self.bin parameter
    def find_sideband_limits(self):
        lower_limits = (self.bins[0], self.bins[0]+30)
        i_lower_limits = self.find_limits_index(lower_limits)
        self.lower_sideband_limits = i_lower_limits
        upper_limits = (self.bins[self.num_bins]-30, self.bins[self.num_bins])
        i_upper_limits = self.find_limits_index(upper_limits)
        self.upper_sideband_limits = i_upper_limits
        return i_lower_limits, i_upper_limits

    #method that does a least square fit to the sideband data to obtain a straight lines parameters (the gradient and
    #y-intercept) that characterizes the background of the data
    def lstsqr_fit_noise(self):
        temp_x = self.bins[self.upper_sideband_limits[0]:self.upper_sideband_limits[1]]
        x_data = np.append(self.bins[self.lower_sideband_limits[0]:self.lower_sideband_limits[1]], temp_x)
        temp_y = self.n[self.upper_sideband_limits[0]:self.upper_sideband_limits[1]]
        y_data = np.append(self.n[self.lower_sideband_limits[0]:self.lower_sideband_limits[1]], temp_y)
        A = np.vstack([x_data, np.ones(len(x_data))]).T
        m, c = np.linalg.lstsq(A, y_data, rcond=None)[0] #calls least square fit function within numpy.linalg
        self.noise_fit = [m, c]
        return m, c

    #method to obtain the number of events that characterize the background of the data from the parameters previously
    #found for the least square fit of the straight line
    def signal_num_noise(self):
        m = self.noise_fit[0]
        c = self.noise_fit[1]
        self.B = np.sum(m*(self.bins[self.signal_limits[0]:self.signal_limits[1]])+c)
        return self.B

    #method to find tha actual number of events that are believed to have been due to Jpsi meson production and not
    #background
    def actual_signal_events(self):
        self.S = self.N - self.B
        return self.S, self.N

    ##Single Gaussian
    #method that returns the output of a gaussian with an exponential decay background from parameters given to it
    def gauss(self, x, F, a, mu, st):
        y = (1-F)*self.single_gaussian.single_gaussian(x, mu, st)+F*self.exponential(x, a)
        return y

    #method that returns the output of the negative natural logarithm of the maximum likelihood of a single gaussian
    #with an exponential decay for background from parameters given to it
    def NLOGL(self, F, a, mu, st):
        a = np.longdouble(a)
        L = -1*np.sum(np.log(self.gauss(self.data, F, a, mu, st)))
        return L

    #Double Gaussian
    #method that returns the output of a double gaussian with an exponential decay background from parameters
    # given to it
    def gauss_two(self, x, F, a, mu_1, st_1, st_2, Q):
        y = (1 - F) * self.double_gaussian.double_gaussian(x, mu_1, st_1, st_2, Q) + F * self.exponential(x, a)
        return y

    #method that returns the output of the negative natural logarithm of the maximum likelihood of a double gaussian
    #with an exponential decay for background from parameters given to it
    def NLOGL_2(self, F, a, mu_1, st_1, st_2, Q):
        a = np.longdouble(a)
        Q = np.longdouble(Q)
        L = -1*np.sum(np.log(self.gauss_two(self.data, F, a, mu_1, st_1, st_2, Q)))
        return L

    ## Crystal Ball
    #method that returns the output of the negative natural logarithm of the maximum likelihood of a crystal ball
    #function with an exponential decay for background from parameters given to it
    def NLOGL_crystalball(self, n, a, mu, st, F, w):
        n = (n)
        a = np.longdouble(a)
        mu = np.longdouble(mu)
        st = np.longdouble(st)
        F = np.longdouble(F)
        w = np.longdouble(w)
        e = (self.data - mu) / st
        x_1 = self.data[(e > -a)]
        x_2 = self.data[(e <= -a)]
        L_1 = float()
        L_2 = float()
        if len(x_1) > 0: #passes data that satifisies first condition in crystal ball function
            L_1 = np.sum((np.log((1-F)*self.crystal_ball.crystal_ball_1(x_1, n, a, mu, st)+F*self.exponential(x_1, w))))
        if len(x_2) > 0: #passes data that satifisies second condition in crystal ball function
            L_2 = np.sum((np.log((1-F)*self.crystal_ball.crystal_ball_2(x_2, n, a, mu, st)+F*self.exponential(x_2, w))))
        L = -1*(L_1+L_2)
        return L

    #method that plots a histogram of the Probability density function with a crystal ball fitted to it from parameters
    #found using a maximum likelihood fit
    def plot_crystal_ball(self, fig_num):
        n = self.parameters_cb[0]
        a = self.parameters_cb[1]
        mu = self.parameters_cb[2]
        st = self.parameters_cb[3]
        F = self.parameters_cb[4]
        w = self.parameters_cb[5]
        exp_y = np.array([])
        crystal_y = np.array([])
        plt.figure(fig_num)
        ax1 = plt.subplot(211)
        #Finds y for the model to be fitted and just for the crystal ball part
        for i in range(0, len(self.bins)-1):
            x = self.bins[i]
            if (x-mu)/st > -a:
                y = (1-F)*self.crystal_ball.crystal_ball_1(x, n, a, mu, st)+F*self.exponential(x, w)
                exp_y = np.append(exp_y, np.array([y]))
                c_y = (1 - F) * self.crystal_ball.crystal_ball_1(x, n, a, mu, st)
                crystal_y = np.append(crystal_y, np.array([c_y]))
            elif (x-mu)/st <= -a:
                y = (1-F)*self.crystal_ball.crystal_ball_2(x, n, a, mu, st)+F*self.exponential(x, w)
                exp_y = np.append(exp_y, np.array([y]))
                c_y = (1-F)*self.crystal_ball.crystal_ball_2(x, n, a, mu, st)
                crystal_y = np.append(crystal_y, np.array([c_y]))
        self.expy_cb = exp_y
        expo_y = F*self.exponential(self.bins[0:self.num_bins], w) #finds y for background exponential decay
        plt.xlabel(self.xlabel)
        plt.ylabel("Probability Density per " + str(round_digits(self.bin_width, 2))+ " "+ self.units)
        plt.title("Probability Density Function with Crystal Ball Function Fit:\n " + self.title)
        u, bins, patches = plt.hist(self.data, self.num_bins, self.limits, density=True, stacked=True) #Note that
        # density and stacked being true ensures histogram is normalized and thus a PDF (Probability Denisty Function)
        ax1.plot(self.bins[0:self.num_bins], self.expy_cb, label="Full Crystal Ball Model Fit")
        ax1.plot(self.bins[0:self.num_bins], crystal_y, label="Crystal Ball Function Part")
        ax1.plot(self.bins[0:self.num_bins], expo_y, label="Background Exponential Decay Fit")
        ax1.legend()
        return self.expy_cb

    #method that plots a histogram of the Probability density function with a single gaussian to it from parameters
    #found using a maximum likelihood fit
    def plot_single_gaussian(self, fig_num):
        F = self.parameters_g[0]
        a = self.parameters_g[1]
        mu = self.parameters_g[2]
        st = self.parameters_g[3]
        plt.figure(fig_num)
        ax1 = plt.subplot(211)
        exp_y = self.gauss(self.bins[0:self.num_bins], F, a, mu, st) #finds y for full model
        gauss_y = (1-F)*self.single_gaussian.single_gaussian(self.bins[0:self.num_bins], mu, st) #finds y for just gaussian part
        expo_y = F*self.exponential(self.bins[0:self.num_bins], a) #finds y for exponential background decay
        self.expy_single = exp_y
        plt.xlabel(self.xlabel)
        plt.ylabel("Probability Density per " + str(round_digits(self.bin_width, 2))+ " "+ self.units)
        plt.title("Probability Density Function with Single Gaussian Fit:\n " + self.title)
        self.prob, bins, patches = plt.hist(self.data, self.num_bins, self.limits, density=True, stacked=True)
        ax1.plot(self.bins[0:self.num_bins], self.expy_single, label="Full Single Gaussian Model Fit")
        ax1.plot(self.bins[0:self.num_bins], gauss_y, label="Single Gaussian Part")
        ax1.plot(self.bins[0:self.num_bins], expo_y, label="Exponential Background Decay")
        ax1.legend()
        return self.expy_single

    #method that plots a histogram of the Probability density function (PDF) with a double gaussian to it from parameters
    #found using a maximum likelihood fit
    def plot_double_gaussian(self, fig_num):
        F = self.parameters_2g[0]
        a = self.parameters_2g[1]
        mu_1 = self.parameters_2g[2]
        st_1 = self.parameters_2g[3]
        st_2 = self.parameters_2g[4]
        Q = self.parameters_2g[5]
        plt.figure(fig_num)
        ax1 = plt.subplot(211)
        print(self.bin_width)
        self.expy_double = self.gauss_two(self.bins[0:self.num_bins], F, a, mu_1, st_1, st_2, Q) #finds y for full double gaussian model
        gauss1_y = (1-F)*Q*self.single_gaussian.single_gaussian(self.bins[0:self.num_bins], mu_1, st_1) #finds y for narrow gaussian
        gauss2_y = (1-F)*(1-Q)*self.single_gaussian.single_gaussian(self.bins[0:self.num_bins], mu_1, st_2) #finds y for wide gaussian
        expo_y = F*self.exponential(self.bins[0:self.num_bins], a) #finds y for exponential background decay
        plt.xlabel(self.xlabel)
        plt.ylabel("Probability Density per " + str(round_digits(self.bin_width, 2))+ " "+ self.units)
        plt.title("Probability Density Function with Double Gaussian Fit\n: "+self.title)
        n, bins, patches = plt.hist(self.data, self.num_bins, self.limits, density=True, stacked=True)
        ax1.plot(self.bins[0:self.num_bins], self.expy_double, label="Full Double Gaussian Model Fit")
        ax1.plot(self.bins[0:self.num_bins], gauss1_y, label="Narrow Gaussian Fit")
        ax1.plot(self.bins[0:self.num_bins], gauss2_y, label="Wide Gaussian Fit")
        ax1.plot(self.bins[0:self.num_bins], expo_y, label="Exponential Background Decay")
        ax1.legend()
        return self.expy_double

    #method that plots the residuals between the PDF and a given y set of data
    def plot_residuals(self, y):
        ax2 = plt.subplot(212)
        residuals = self.prob - y
        plt.xlabel(self.xlabel)
        plt.ylabel(str("Residuals of " + self.ylabel))
        plt.title(str("Plot of the Residuals versus "+self.xlabel))
        ax2.scatter(self.bins[0:self.num_bins], residuals)
        plt.tight_layout()