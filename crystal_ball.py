#Imports Modules needed later
import numpy as np
import scipy.special as special

#class for a crystal ball function
class Crystal_ball():
    #Returns the A parameter that is defined below by the equation and parameters given
    def A(self, n, a):
        temp = (n/np.abs(a))
        y = np.longdouble(np.exp((-(np.abs(a)**2)/2))* (np.sign(temp)*np.abs(temp) **n))
        return y

    #Returns the _B parameter that is defined below by the equation and parameters given
    def _B(self, n, a):
        y = np.longdouble((n/np.abs(a))-np.abs(a))
        return y

    #Returns the C parameter that is defined below by the equation and parameters given
    def C(self, n, a):
        y = np.longdouble((n/np.abs(a))*(1/(n-1))*(np.exp(-(np.abs(a)**2)/2)))
        return y

    #Returns the D parameter that is defined below by the equation and parameters given
    def D(self, a):
        y = np.longdouble((np.sqrt(np.pi/2))*(1+special.erf(np.abs(a)/(np.sqrt(2)))))
        return y

    #Returns the Normalization parameter that is defined below by the equation and parameters given
    def _N(self, st, n, a):
        y = np.longdouble(1/(st*(self.C(n,a) +self.D(a))))
        return y

    #method that returns the output from a the first condition of the crystal ball function at x
    def crystal_ball_1(self, x, n, a, mu, st):
        e = np.longdouble((x-mu)/st)
        y = np.longdouble(self._N(st, n, a)*np.exp((-(e**2)/2)))
        return y

    #method that returns the output from a the second condition of the crystal ball function at x
    def crystal_ball_2(self, x, n, a, mu, st):
        e = np.longdouble((x-mu)/st)
        temp = np.float64((self._B(n, a)-e))
        y = np.float64(self._N(st, n, a)*(self.A(n, a)*(np.sign(temp) * 1 / (np.abs(temp) ** (n)))))
        return y