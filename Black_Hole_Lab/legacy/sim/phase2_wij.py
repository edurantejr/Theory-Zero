import numpy as np
w = np.loadtxt(r"C:\path\to\phase2_wij.csv", delimiter=",")  # or .txt
np.save(r"C:\Users\lcpld\Documents\Theory_Zero_Project\Black_Hole_Lab\refs\phase2_wij.npy", w)
