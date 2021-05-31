import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['font.size'] = 28
plt.rcParams['ytick.labelsize'] = 28
data = pd.read_csv('./catas/4FGLDR2comparison.csv', quoting=2,usecols=['diffBDT2'])

data.hist(bins=50)
plt.title("Boosted Decision Trees")
plt.xlabel("Normalized difference of Probability")
plt.ylabel("Frequency")
plt.show()


