import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['xtick.labelsize'] = 32
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['font.size'] = 28
plt.rcParams['ytick.labelsize'] = 28
data = pd.read_csv('./catas/4FGL-DR2_2Class_Catalog_AllSamplings.csv', quoting=2,usecols=['diffPSR_LR'])

data.hist(bins=50)
plt.title("Logistic regression, 4FGL-DR2, PSR-like probabilities (2-class)")
plt.xlabel(xlabel=r'$\frac{P_O-P_S}{max(\sigma_O,\sigma_S)}$',fontsize=34)
plt.xlim(-3,3)
#plt.ylabel("Frequency")
plt.show()


