#Imports Modules needed later
from gaussian import Gaussian

#class for a double gaussian
class Double_gaussian(Gaussian): #inherits gaussian class
    #method that returns the output from a double gaussian curve at x
    def double_gaussian(self, x, mu_1, mu_2, st_1, st_2, Q):
        y = (Q*self.single_gaussian(x, mu_1, st_1)+(1-Q)*self.single_gaussian(x, mu_2, st_2))
        return y