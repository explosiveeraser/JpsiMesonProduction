#Imports Modules needed later
import numpy as np

#Function that obtains data from bin file and splits it into a 2d array with 7 different types of data in each index
def get_data(file_name):
    f = open(file_name, "r")
    b = np.fromfile(f, dtype=np.float32)
    ncol = 6
    nevent = len(b) // 7
    x = np.array(np.split(b, nevent))
    np.set_printoptions(suppress=False, threshold=10000)
    f.close()
    return x, nevent