import numpy as np
from random import shuffle, sample
from astropy.io import fits
from numpy import vstack
from sklearn import preprocessing
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import pandas
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot

import plotting_dima


pyplot.rcParams['xtick.labelsize'] = 18
pyplot.rcParams['axes.labelsize'] = 18
pyplot.rcParams['axes.titlesize'] = 28
pyplot.rcParams['font.size'] = 21
pyplot.rcParams['ytick.labelsize'] = 18
pyplot.rcParams['lines.linewidth'] = 4
pyplot.rcParams['lines.markersize'] = 12


dataframe = pandas.read_csv("./files/result_3fglassocnewfeat_nn_neurons_2layers_multi_oversampled.csv", header=None)
dataset1 = dataframe.values
#dataframe2 = pandas.read_csv("./files/result_3fglassocnewfeat_rf2.csv", header=None)
#dataset2=dataframe2.values

fig,ax=plt.subplots()
numi=dataset1[1]
'''
valscore31=dataset2[2]
valscore34=dataset2[3]
valscore32=dataset2[4]
'''
valscore3=dataset1[2]
valscore4=dataset1[3]
valscore12=dataset1[4]
valscore22=dataset1[5]
'''
plt.plot(numi, valscore31, 'o--',marker='x')
plt.plot(numi, valscore34, 'b',marker='.')
plt.plot(numi, valscore32, 'c-',marker='>')
'''
plt.plot(numi, valscore3, 'g-',marker='D')
plt.plot(numi, valscore4, 'p-.',marker='o')
plt.plot(numi, valscore12, 'r--',marker='>')
plt.plot(numi, valscore22, 'y:',marker='<')


ax.set_xlabel('Number of Neurons in second layer',fontsize='xx-large')
ax.set_ylabel('Testing Score',fontsize='xx-large')
plt.yticks(fontsize='large')
#plt.yticks(np.arange(92,99,step=1))
#plt.xticks(fontsize='large')
#ax.set_zlabel('Validation score')
plt.legend(["LBFGS, Tanh","LBFGS, Relu","Adam, Tanh","Adam, Relu"],title='Number of Epochs: 600')
#plt.legend(["5 Trees","20 Trees","100 Trees","500 Trees"])#"50 Trees","100 Trees","200 Trees"])

#plt.legend(["LBFGS","SAG","SAGA"])
#plt.legend(["Tol = 0.001","Tol = 1","Tol = 10"])

ax.set_title('Neural Network',fontsize='xx-large')
plt.show()
