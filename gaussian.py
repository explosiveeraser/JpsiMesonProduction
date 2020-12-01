#Imports Modules needed later
import numpy as np

#class for gaussian curve
class Gaussian():
    #method that returns the output from a gaussian curve at x
    def single_gaussian(self, x, mu, st):
        y = (1 / (st * np.sqrt(2 * np.pi))) * np.exp(-(0.5) * ((x - mu) / st) ** 2)
        return y