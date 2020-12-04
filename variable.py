#Imports Modules needed later
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from iminuit import Minuit
from math import log10, floor

#Function that rounds x
def round_digits(x, n):
    return round(x, -int(floor(log10(abs(x))))+n)


#Creates a class that is used to analyse the variables in the Jpsi data from the Jpsi class
class Variable():
    # Method to initialize the class parameters to be used later for analysis
    def __init__(self, data, limits, num_bins, title, xlabel, ylabel, file_name, fig_num, units):
        self.data = data
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
        self.prob = []
        self.bin_width = (np.max(self.data)-np.min(self.data))/self.num_bins
        self.units = units

    #method to plot a histogram of the variables data
    def plot_hist(self):
        self.fig = plt.figure(self.fig_num)
        plt.xlabel(self.xlabel)
        plt.title(self.title)
        self.n, self.bins, self.patches = plt.hist(self.data, self.num_bins, self.limits)
        plt.ylabel(self.ylabel + "per " + str(round_digits(self.bin_width, 2)) + " " + self.units)
        plt.savefig(self.file_name)

    #method that returns the output of a exponential decay that is characterized by a parameter a (that is given)
    #this exponential decay is also normalized between the minimum and maximum of the data given
    def exponential(self, x, a):
        y = np.longdouble((-1 * a / (np.exp(-1 * a * np.max(self.data)) - np.exp(-1 * a * np.min(self.data)))) * np.exp(-1 * a * x))
        return y

    ##method that finds the parameters from a given function and initial values utilizing the iMinuit module for
    #minimization
    def maximum_likelihood_est(self, func, kwargs):
        lik_model = Minuit(func, **kwargs)
        lik_model.migrad()
        parameters = lik_model.values
        print(parameters)
        print(lik_model.errors)
        errors = lik_model.errors
        return parameters, errors