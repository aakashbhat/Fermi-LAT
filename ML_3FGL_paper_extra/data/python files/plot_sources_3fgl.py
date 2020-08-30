print(__doc__)


# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause
import pandas
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import matplotlib.colors as colors


plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['font.size'] = 23
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['lines.markersize'] = 13



#Training Fata:
se=4
np.random.seed(se)
dataframe = pandas.read_csv("./catas/3FGL_unassocvs4FGLassoc_AGN&PSR_catalog_unweighted.csv", header=None)
dataset1 = dataframe.values 
#np.random.shuffle(dataset1[1:])
X = dataset1[1:,59:65]
print(X)
#Y = dataset[1:1933,5]
y = dataset1[1:,3:5].astype(float)
print(y)
#dataset=dataset[:,:5]
num=[]
num2=[]


c1_mixed = [i for i in range(278) if X[i,0] =='true']
c1_class_psr = [i for i in range(278) if X[i,1] =='true']
c1_miss_psr = [i for i in range(278) if X[i,2] =='true']
c1_mixed_agn = [i for i in range(278) if X[i,3] =='true']
c1_class_agn = [i for i in range(278) if X[i,4] =='true']
c1_missclass_agn = [i for i in range(278) if X[i,5] =='true']

print(c1_miss_psr)
trainc1_color = 'green'
trainc1_marker = 's'
testc1_color = 'yellow'
testc1_marker = 'd'
    
trainc2_color = 'blue'
trainc2_marker = '^'
testc2_color = 'magenta'
testc2_marker = 'v'

trainc3_color = 'red'
trainc3_marker = '>'
testc3_color = 'black'
testc3_marker = '<'

X=y
print(X)
alpha=0.7

fig1,ax2=plt.subplots()

ax2.scatter(y[c1_mixed, 0], y[c1_mixed, 1], c=trainc2_color, alpha=alpha,
                   marker=trainc2_marker, edgecolors='k', label='Mixed PSR')
# Testing points for class 2
ax2.scatter(y[c1_class_psr, 0], y[c1_class_psr, 1], c=testc2_color, alpha=alpha,
                   marker=testc2_marker, edgecolors='k', label='Classified PSRs')
    # Training points for class 1
ax2.scatter(y[c1_miss_psr, 0], y[c1_miss_psr, 1], c=trainc1_color, alpha=alpha,
                   marker=trainc1_marker, edgecolors='k', label='Misclassified PSRs')
        # Testing points for class 1
ax2.scatter(y[c1_mixed_agn, 0], y[c1_mixed_agn, 1], c=testc1_color, alpha=alpha,
                   marker=testc1_marker, edgecolors='k', label='Mixed AGNs')
ax2.scatter(y[c1_class_agn, 0], y[c1_class_agn, 1], c=trainc3_color, alpha=alpha,
                   marker=trainc3_marker, edgecolors='k', label='Classified AGNs')
        # Testing points for class 2
ax2.scatter(y[c1_missclass_agn, 0], y[c1_missclass_agn, 1], c=testc3_color, alpha=alpha,
                   marker=testc3_marker, edgecolors='k', label='Misclassified AGNs')
        # Training points for class 1

h=0.02
ax2.legend(loc=1)
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5   #Define minimumm and maximum of axes
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
#print(clf.n_layers_)
ax2.set_xlim(xx.min(), xx.max())
ax2.set_title('Predictions for 3FGL Unassociated Data')
#ax2.set_ylim(yy.min(), yy.max())
ax2.set_xlabel('Spectral Index')
ax2.set_ylabel('Ln(Significant_Curvature)')
ax2.set_ylim((-4,3))
ax2.set_xlim((1,4))

#ax.set_xticks(np.arange(-5,3,step=1))
 #       ax.set_yticks(())
  #      if ds_cnt == 0:
   #         ax.set_title(name)
#ax2.text(xx.max() , yy.max() + .3, ('%.2f' % score).lstrip('0'),
#               size=15, horizontalalignment='right')    
plt.show()
print(c1_mixed)
