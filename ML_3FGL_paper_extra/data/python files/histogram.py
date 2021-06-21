import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['font.size'] = 28
plt.rcParams['ytick.labelsize'] = 28
data = pd.read_csv('./catas/3FGLcomparison_3CLASS.csv', quoting=2,usecols=['diffBDT'])

data.hist(bins=50)
plt.title("Neural Network")
plt.xlabel(xlabel=r'$\frac{P_O-P_S}{max(\sigma_O,\sigma_S)}$')
plt.xlim(-3,3)
#plt.ylabel("Frequency")
plt.show()


