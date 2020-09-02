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

'''
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['font.size'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['lines.markersize'] = 12
'''
import plotting_dima
plotting_dima.setup_figure_pars()

#Training Fata:
se=4
np.random.seed(se)
dataframe = pandas.read_csv("./files/3fgl_associated_AGNandPSR.csv", header=None)
dataset1 = dataframe.values 
np.random.shuffle(dataset1[1:])



X = dataset1[1:,0:10].astype(float)
print(X)
Y = dataset1[1:,10]
encoder = preprocessing.LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)



h = .02  # step size in the mesh




#X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
 #                         random_state=1, n_clusters_per_class=1)

#Making datsets, these were initially with different features and not the same:
X1=dataset1[1:,2:4]
print(dataset1[0,2:4])
y1=Y
linearly_separable = (X1, y1)
print(linearly_separable)
datasets = [linearly_separable]
X, y = datasets[0]



X = StandardScaler(with_mean=False,with_std=False).fit_transform(X)
X_train, X_test, y_train, y_test = \
 train_test_split(X, y, test_size=.3, random_state=0)       #Split into training and validation

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5   #Define minimumm and maximum of axes
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax1 = plt.subplot()
    #ax.legend(('Red=AGN', 'Blue=PSR'))
'''    
    if ds_cnt == 1:
        ax.set_title("Input data")
        ax.set_ylabel("Variability")
        ax.set_xlabel("Galactic Latitude")
        #ax.legend(['AGN','PSR'])
    elif ds_cnt== 0:
        ax.set_xlabel("Spectral Index")
        ax.set_ylabel("Significant Curvature")
        ax.set_xticks(np.arange(0,3,step=0.2))
        ax.set_yticks(())
    else :
        ax.set_xlabel("Flux_Density")
        ax.set_ylabel("Uncertainity on Energy Flux")
'''        
    # Plot the training points for input plot, Not really necessary
ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    
    # Plot the testing points
ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    
ax1.set_xlim(xx.min(), xx.max())
ax1.set_ylim(yy.min(), yy.max())
ax1.set_xlabel('1')
ax1.set_ylabel('2')
ax1.set_title('Input Data')

#plt.show()

#Main part of code:
c1_train_inds = [i for i in range(len(y_train)) if y_train[i] > 0.5]
c1_test_inds = [i for i in range(len(y_test)) if y_test[i] > 0.5]
c2_train_inds = [i for i in range(len(y_train)) if y_train[i] <= 0.5]
c2_test_inds = [i for i in range(len(y_test)) if y_test[i] <= 0.5]

fig1,ax2 = plt.subplots()

trainc1_color = 'green'
trainc1_marker = 's'
# for some reason squares look visually larger than triangles
# this is a hack to make the a bit smaller
trainc1_marker_size = plt.rcParams['lines.markersize'] - 0.5 

testc1_color = 'yellow'
testc1_marker = 'd'
    
trainc2_color = 'blue'
trainc2_marker = '^'
testc2_color = 'magenta'
testc2_marker = 'v'

#ml_str = 'bdt_20_2'
#ml_str = 'bdt_100_6'
#ml_str = 'lr_200_lbfgs'
#ml_str = 'nn_adam_10_tanh_50_final'
#ml_str = 'nn_adam_10_tanh_300_final'
#ml_str = 'rf_20_2_final'
ml_str = 'rf_50_6_final'



#Choose classifier:
if ml_str == 'rf_20_2_final':
    clf=RandomForestClassifier(max_depth=2, n_estimators=20, class_weight='balanced',oob_score=True)
    info1 = "Trees: 20"
    info2 = "Maximum Depth: 2"
    title = 'Random Forests'
elif ml_str == 'rf_50_6_final':
    clf=RandomForestClassifier(max_depth=6, n_estimators=50, class_weight='balanced', oob_score=True)
    info1 = "Trees: 50"
    info2 = "Maximum Depth: 6"
    title = 'Random Forests'
elif ml_str == 'nn_adam_10_tanh_50_final':
    clf= MLPClassifier(max_iter=50, hidden_layer_sizes=(2,), activation='tanh', solver='adam')
    info1 = "Epochs: 50"
    info2 = "Solver: Adam"
    title = 'Neural networks'
elif ml_str == 'nn_adam_10_tanh_300_final':
    clf= MLPClassifier(max_iter=300, hidden_layer_sizes=(2,), activation='tanh', solver='adam')
    info1 = "Epochs: 300"
    info2 = "Solver: Adam"
    title = 'Neural networks'
elif ml_str == 'bdt_100_6':
    clf=GradientBoostingClassifier(n_estimators=100, learning_rate=0.3, max_depth=6, random_state=0)
    info1 = "Trees: 100"
    info2 = "Maximum Depth: 6"
    title = 'Boosted Decision Trees'
elif ml_str == 'bdt_20_2':
    clf=GradientBoostingClassifier(n_estimators=20, learning_rate=0.3, max_depth=2, random_state=0)
    info1 = "Trees: 20"
    info2 = "Maximum Depth: 2"
    title = 'Boosted Decision Trees'
elif ml_str == 'lr_200_lbfgs':
    clf=LogisticRegression(max_iter=200, C=0.1, solver='lbfgs').fit(X_train,y_train)
    info1 = "Iterations: 200"
    info2 = "Solver: LBFGS"
    title = 'Logistic regression'
clf.fit(X_train, y_train)

lenth=len(X_test)
score = clf.score(X_test, y_test)       #Score of our classifier
i=0
levels = np.arange(0., 1.01, 0.1)
cm = plt.cm.GnBu_r
norm = colors.Normalize(vmin=0, vmax=0.9)

X_test_spec_agn=[]
X_test_sig_agn=[]
y_test_agn=[]
X_test_spec_psr=[]
X_test_sig_psr=[]
y_test_psr=[]
for i in range(lenth):
    if y_test[i]==0:
        X_test_spec_agn.append(X_test[i,0])
        X_test_sig_agn.append(X_test[i,1])
        y_test_agn.append(y_test[i])
    else:
        X_test_spec_psr.append(X_test[i,0])
        X_test_sig_psr.append(X_test[i,1])
        y_test_psr.append(y_test[i])
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
#if hasattr(clf, "decision_function"):
#    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#else:
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
print(y_test_psr)
# Put the result into a color plot
Z = Z.reshape(xx.shape)
cs=ax2.contourf(xx, yy, Z, cmap=cm, alpha=.8,levels=levels)
fig1.colorbar(cs,ax=ax2,shrink=0.9)
        # Plot the training points
alpha=0.9

# Training points for class 2
ax2.scatter(X_train[c2_train_inds, 0], X_train[c2_train_inds, 1], c=trainc2_color, alpha=alpha,
                   marker=trainc2_marker, edgecolors='k', label='AGN training')
        # Testing points for class 2
ax2.scatter(X_test[c2_test_inds, 0], X_test[c2_test_inds, 1], c=testc2_color, alpha=alpha,
                   marker=testc2_marker, edgecolors='k', label='AGN testing')
        # Training points for class 1
ax2.scatter(X_train[c1_train_inds, 0], X_train[c1_train_inds, 1], c=trainc1_color, alpha=alpha,
                   marker=trainc1_marker, s=trainc1_marker_size**2, edgecolors='k', label='PSR training')
        # Testing points for class 1
ax2.scatter(X_test[c1_test_inds, 0], X_test[c1_test_inds, 1], c=testc1_color, alpha=alpha,
                   marker=testc1_marker, edgecolors='k', label='PSR testing')

ax2.legend()
#ax2.text(0.02,-1.0,"Solver: Adam",fontsize=23)
ax2.text(0.02,-1.3,info1)
ax2.text(0.02,-1.6,info2)
ax2.set_xlim(xx.min(), xx.max())
ax2.set_title(title)
#ax2.set_ylim(yy.min(), yy.max())
ax2.set_xlabel('Spectral Index')
ax2.set_ylabel('Ln(Significant_Curvature)')
ax2.set_ylim((-2,5))
#ax.set_xticks(np.arange(-5,3,step=1))
 #       ax.set_yticks(())
  #      if ds_cnt == 0:
   #         ax.set_title(name)
ax2.text(0.02 , -1.9, ('Testing Score: %.2f' % score).lstrip('0'))
 #       i += 1


#plt.tight_layout()
#plt.show()
fn = 'plots/%s.pdf' % ml_str
print('save plot to file')
print(fn)
plt.savefig(fn)

