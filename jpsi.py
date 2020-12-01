#Imports Modules needed later
from variable import Variable
from gaussian_variable import Gaussian_variable

#creates class to hold data and analysis data from jpsi meson production
class Jpsi():
    # Method to initialize the class parameters to be used later for analysis
    def __init__(self, x, nevent, fig_nums=(1,2,3,4,5,6,7)):
        self.nevent = nevent
        self.xmass = Gaussian_variable(self.create_variable(x, 0), (0,0), 250, "Candidates versus Invariant Mass", "Invariant Mass (MeV/c^2)", "Candidates", "invariant_mass_hist.png", fig_nums[0]) #invariant mass
        self.tmomentum = Variable(self.create_variable(x, 1), (0.0, 8000.0), 100, "Candidates versus Transverse Momentum", "Transverse Momentum (MeV/c)", "Candidates", "transverse_momentum_hist.png", fig_nums[1]) #transverse momentum
        self.rapidity = Variable(self.create_variable(x, 2), (1.8, 7.0), 100, "Candidates versus Rapidity", "Rapidity", "Candidates", "rapidity_hist.png", fig_nums[2]) #rapidity
        self.chiSqr = Variable(self.create_variable(x, 3), (0, 1.5), 100, "Candidates versus Geometric Vertex \nof Dimuon Candidate", "Geometric Vertex of Dimuon Candidate", "Candidates", "geomtric_vertex_hist.png", fig_nums[3]) #geometric vertex of dimuon candidate
        self.minTMomentum = Variable(self.create_variable(x, 4), (0.0, 2600.0), 100, "Candidates versus Minimum Transverse Momentum", "Minimum Transverse Momentum (MeV/c)", "Candidates", "minimum_transverse_momentum_hist.png", fig_nums[4]) #Minimum transverse momentum
        x_prob = "Minimum of a variable (ProbNNmu) that characterizes how well the two\n tracks forming the candidate match the hypothesis of being muons."
        self.ProbNNum =  Variable(self.create_variable(x, 5), (0,0), 100, str("Candidates versus "+x_prob), x_prob, "Candidates", "ProbNNum_hist.png", fig_nums[5]) #ProbNNmu
        self.impactChi = Variable(self.create_variable(x, 6), (0.0, 500.0), 100, "Candidates versus Minimum Impact Parameter of two Muons", "Minimum Impact Parameter", "Candidates", "impact_parameter_hist.png", fig_nums[6]) #Miminum Impact parameter
        xmass_limits = (min(self.xmass.data), max(self.xmass.data))
        ProbNNum_limits = (min(self.ProbNNum.data), max(self.ProbNNum.data))
        self.xmass.limits = xmass_limits
        self.ProbNNum.limits = ProbNNum_limits

    #method returns the all the data from the jth variable in the 7 index long 2d array that contains all the data
    # for analysing
    def create_variable(self, x, j):
        variable = x[:,j]
        return variable

    #method that plots all the histograms of the 7 different variables to be seen by the user
    def plot_hists(self):
        self.xmass.plot_hist()
        self.tmomentum.plot_hist()
        self.rapidity.plot_hist()
        self.chiSqr.plot_hist()
        self.minTMomentum.plot_hist()
        self.ProbNNum.plot_hist()
        self.impactChi.plot_hist()

    #method that obtains the number of background and actual signal events in the invariant mass histogram
    def xmass_find_B_and_S(self):
        self.xmass.find_mean()
        self.xmass.find_signal_limits()
        self.xmass.num_events_signal_width()
        self.xmass.find_sideband_limits()
        self.xmass.lstsqr_fit_noise()
        B = self.xmass.signal_num_noise()
        S, N =  self.xmass.actual_signal_events()
        return S, B, N

    #method that finds the parameters and parameter errors of a gaussian model for the probability density function
    # (PDF) histogram of the invariant mass and the plots the PDF with a fit for the gaussian model
    def fit_single_gaussian(self, fig_num, kwargs=dict(F=0.63, a=1.e-3, mu=3096., st=14., limit_F=(0., 1.))):
        self.xmass.parameters_g, self.xmass.errors_g = self.xmass.maximum_likelihood_est(self.xmass.NLOGL, kwargs)
        y = self.xmass.plot_single_gaussian(fig_num)
        self.xmass.plot_residuals(y)

    # method that finds the parameters and parameter errors of a double gaussian model for the probability density function
    # (PDF) histogram of the invariant mass and the plots the PDF with a fit for the double gaussian model
    def fit_double_gaussian(self, fig_num, kwargs=dict(F=0.7, a=1.e-3, mu_1=3096., mu_2=3096., st_1=10., st_2=40., Q=0.5, limit_F=(0., 1.), limit_Q=(0., 1.))):
        self.xmass.parameters_2g, self.xmass.errors_2g = self.xmass.maximum_likelihood_est(self.xmass.NLOGL_2, kwargs)
        y = self.xmass.plot_double_gaussian(fig_num)
        self.xmass.plot_residuals(y)

    # method that finds the parameters and parameter errors of a crystal ball function model for the
    # probability density function (PDF) histogram of the invariant mass and the plots the PDF with a fit for the
    # crystal ball model
    def fit_crystal_ball(self, fig_num, kwargs=dict(n=-0.4, a=3., mu=3097.246, st=14.147, F=0.6315, w=1.322e-3, limit_F=(0., 1.), limit_a=(0., 8.), limit_n=(-10., 10.))):
        self.xmass.parameters_cb, self.xmass.errors_cb = self.xmass.maximum_likelihood_est(self.xmass.NLOGL_crystalball, kwargs)
        y = self.xmass.plot_crystal_ball(fig_num)
        self.xmass.plot_residuals(y)

    #method that fits 3 types of models to the invariant mass PDF histogram and fits it to the data
    def fit_all_models(self, fig_nums, list_kwargs):
        self.fit_single_gaussian(fig_nums[0], list_kwargs[0])
        self.fit_double_gaussian(fig_nums[1], list_kwargs[1])
        self.fit_crystal_ball(fig_nums[2], list_kwargs[2])


