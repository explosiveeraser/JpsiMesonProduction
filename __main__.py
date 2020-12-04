# Imports Modules needed later
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import readbin as rb
from jpsi import Jpsi


# Function returns filtered out data with an impact parameter less than or equal to 4.0 and length of filtered data
def filter_data_with_impact_condition(x):
    filtered_data = x[x[:, 6] > 4.]
    length = len(filtered_data)
    return filtered_data, length


# Function returns filtered out data with an impact parameter less than or equal to 4.0
# and ProbNNmu of less than or equal to 0.50 and length of filtered data
def filter_more(x):
    x1 = x[x[:, 6] > 4.]
    x2 = x1[x1[:, 5] > 0.5]
    filtered_data = x2[x2[:, 4] < 14000.]
    length = len(filtered_data)
    return filtered_data, length


#Function returns an array containing only the transverse momentum and rapidity of the data and also filters out the
#indices with transverse momentum more than or equal to 10000.0
def filter_data_mass_pt_eta(x):
    data = np.array([x[:, 0], x[:, 1], x[:, 2]])
    data = data[:, data[1] < 10000.]
    data = data[:, data[2] < 6.2]
    return data


#Passes file name into function to obtain the data of the file plus the number of events recorded in the data
x, nevent = rb.get_data("jpsi.bin")

print(nevent)

#Defines object with Jpsi class and passes it x (data) and nevent (number of events)
jpsi = Jpsi(x, nevent)

#Plots histograms of variables in jpsi object
jpsi.plot_hists()

#Uses find_mean() method to find the mean invariant mass in the data and prints the result
jpsi.xmass.find_mean()
print(jpsi.xmass.mean)

#Pass difference between 3110 and 3080 to set the Full Width Half Maximum after visual inspection of invariant
#mass histogram
print(jpsi.xmass.set_FWHM(3110 - 3080))

#Prints the result of the find_resolution method, that finds the resolution of the invariant mass
print(jpsi.xmass.find_resolution())

#Prints the Number of events with the number of events that include the signal and the number of events that include
#the background by calling xmass_find_B_and_S() method
S, B, N = jpsi.xmass_find_B_and_S()
print("Number of Events in Signal Region=" + str(N) + "|S=" + str(S) + "|B=" + str(B))

##Single Gaussian
#Sets the initial guesses for the parameters of the single gaussian with exponential background fit for the invariant
#mass
kwargs = dict(F=0.63, a=1.e-3, mu=3096., st=14., limit_F=(0., 1.), limit_mu=(np.min(jpsi.xmass.data),
                                                                             np.max(jpsi.xmass.data)))

##Double Gaussian
#Sets the initial guesses for the parameters of the double gaussian with exponential background fit for the invariant
#mass
kwargs_2 = dict(F=0.7, a=1.e-3, mu_1=3096., st_1=10., st_2=20., Q=0.5, limit_F=(0., 1.), limit_Q=(0., 1.),
                limit_mu_1=(np.min(jpsi.xmass.data), np.max(jpsi.xmass.data)))

##Crystal Ball
#Sets the initial guesses for the parameters of the crystal ball with exponential background fit for the invariant
#mass
#Note for crystal ball model w characterizes the background slope.
kwargs_cb = dict(n=1.1, a=3., mu=3097.246, st=14.147, F=0.6315, w=1.322e-3, fix_a=False, fix_mu=False, fix_st=False,
                 fix_F=False, fix_w=True, limit_F=(0., 1.), limit_a=(0., 8.), limit_n=(-10., 10.),
                 limit_mu=(np.min(jpsi.xmass.data), np.max(jpsi.xmass.data)))

#Creates a list of all the initial guesses for the various models
kwargs_list = [kwargs, kwargs_2, kwargs_cb]

#Calls a method in jpsi to fit the invariant mass with the 3 different models onto a PDF (Probability Density Function)
#of the histogram
jpsi.fit_all_models([10, 11, 12], kwargs_list)

##Data filtered to all have impact parameter >4.0 from this point until said otherwise
#Passes the data into a function that returns the data that has an impact parameter greater than 4.0 and also returns
#the length of the filtered data
filtered_data, filtered_length = filter_data_with_impact_condition(x)

#Creates an object of class Jpsi with new filtered data
new_jpsi = Jpsi(filtered_data, int(filtered_length), (15, 16, 17, 18, 19, 20, 21))

#Uses method to plot histograms of the filtered data contained in the new_jpsi object
new_jpsi.xmass.plot_hist()

#Prints the Number of events with the number of events that include the signal and the number of events that include
#the background by calling xmass_find_B_and_S() method from the filtered data (filtered_data)
new_S, new_B, new_N = new_jpsi.xmass_find_B_and_S()
print("Number of Events in Signal Region=" + str(new_N) + "|S=" + str(new_S) + "|B=" + str(new_B))

##Single Gaussian
#Sets the initial guesses for the parameters of the single gaussian with exponential background fit for the invariant
#mass
new_kwargs = dict(F=0.4, a=1.e-3, mu=3096., st=10., limit_F=(0., 1.), limit_mu=(np.min(new_jpsi.xmass.data),
                                                                                np.max(new_jpsi.xmass.data)))

##Double Gaussian
#Sets the initial guesses for the parameters of the double gaussian with exponential background fit for the invariant
#mass
#Note for crystal ball model w characterizes the background slope.
new_kwargs_2 = dict(F=0.3, a=1.e-3, mu_1=3096.,  st_1=10., st_2=40., Q=0.5, limit_F=(0., 1.),
                    limit_Q=(0., 1.), limit_mu_1=(np.min(new_jpsi.xmass.data), np.max(new_jpsi.xmass.data)))

##Crystal Ball
#Sets the initial guesses for the parameters of the crystal ball with exponential background fit for the invariant
#mass
new_kwargs_cb = dict(n=1.1, a=3., mu=3097.246, st=14.147, F=0.6315, w=1.322e-3, fix_a=False, fix_mu=False,
                     fix_st=False, fix_F=False, fix_w=False, limit_F=(0., 1.), limit_a=(0., 8.), limit_n=(-10., 10.),
                     limit_mu=(np.min(jpsi.xmass.data), np.max(jpsi.xmass.data)))

#Makes a list of the initial guesses for each of the models to be fitted
new_jpsi_kwargs = [new_kwargs, new_kwargs_2, new_kwargs_cb]

#Calls a method in object to fit all the models onto a PDF histogram
new_jpsi.fit_all_models([23, 24, 25], new_jpsi_kwargs)

##Data filtered to all have impact parameter >4.0 and ProbNNMu >0.6 from this point until said otherwise
#Passes the data into a function that returns the data that has an impact parameter greater than 4.0 and
#ProbNNmu > 0.6, and also returns the length of the filtered data
data2, data2_length = filter_more(x)

#prints length of filtered data
print(data2_length)

#Creates an object of class Jpsi of the newly filtered data (data2)
jpsi2 = Jpsi(data2, data2_length, (30, 31, 32, 33, 34, 35, 36))

#Plots histograms of the data from data2
jpsi2.plot_hists()

#Prints the Number of events with the number of events that include the signal and the number of events that include
#the background by calling xmass_find_B_and_S() method from the filtered data (data2)
S2, B2, N2 = jpsi2.xmass_find_B_and_S()
print("Number of Events in Signal Region=" + str(N2) + "|S=" + str(S2) + "|B=" + str(B2))

##Single Gaussian
#Sets the initial guesses for the parameters of the single gaussian with exponential background fit for the invariant
#mass
jpsi2_kwargs = dict(F=0.4, a=1.e-3, mu=3096., st=10., limit_F=(0., 1.), limit_mu=(np.min(jpsi2.xmass.data),
                                                                                  np.max(jpsi2.xmass.data)))

##Double Gaussian
#Sets the initial guesses for the parameters of the double gaussian with exponential background fit for the invariant
#mass
#Note for crystal ball model w characterizes the background slope.
jpsi2_kwargs_2 = dict(F=0.3, a=1.e-4, mu_1=3096., st_1=10., st_2=20., Q=0.5, limit_F=(0., 1.),
                      limit_Q=(0., 1.), limit_mu_1=(np.min(jpsi2.xmass.data), np.max(jpsi2.xmass.data)))

##Crystal Ball
#Sets the initial guesses for the parameters of the crystal ball with exponential background fit for the invariant
#mass
jpsi2_kwargs_cb = dict(n=2., a=5., mu=3097.246, st=14.147, F=0.3, w=1.322e-3, fix_a=False,
                       fix_mu=False, fix_st=False, fix_F=False, fix_w=False, limit_F=(0., 1.), limit_a=(0., 8.),
                       limit_n=(-10., 10.), limit_mu=(np.min(jpsi2.xmass.data), np.max(jpsi2.xmass.data)))

##Makes a list of the initial guesses for each of the models to be fitted
jpsi2_kwargs_list = [jpsi2_kwargs, jpsi2_kwargs_2, jpsi2_kwargs_cb]

#Calls a method in object to fit all the models onto a PDF histogram
jpsi2.fit_all_models([37, 38, 39], jpsi2_kwargs_list)

##Transverse Momentum and Rapidity Analysis
#Function pass data to obtain the transverse momentum less than 10000.0MeV/c and rapidity to be plotted later
pt_eta = filter_data_mass_pt_eta(x)

#Initialize figure 45 to be displayed later
fig = plt.figure(45)

#
print(len(pt_eta))
#plots a 2d histogram of the transverse momentum versus the invariant mass
ax1 = plt.subplot(141)
plt.hist2d(pt_eta[0], pt_eta[1], bins=100)
plt.xlabel("Invariant Mass (MeV/c^2)")
plt.ylabel("Transverse Momentum (MeV/c)")
plt.title("Transverse Momentum versus Invariant Mass Histogram")
plt.colorbar()

#plots a 2d histogram of the rapidity versus the invariant mass
ax2 = plt.subplot(142)
plt.hist2d(pt_eta[0], pt_eta[2], bins=100)
plt.xlabel("Invariant Mass (MeV/c^2)")
plt.ylabel("Rapidity")
plt.title("Rapidity versus Invariant Mass Histogram")
plt.colorbar()

##plots a 2d histogram of the rapidity versus the transverse momentum
ax3 = plt.subplot(143)
n = plt.hist2d(pt_eta[1], pt_eta[2], bins=100)
plt.xlabel("Transverse Momentum (MeV/c)")
plt.ylabel("Rapidity")
plt.title("Rapidity versus Transverse Momentum Histogram")
plt.colorbar()
plt.tight_layout()


# plot and show plots
plt.show()
