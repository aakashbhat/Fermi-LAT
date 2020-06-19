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


plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['font.size'] = 23
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['lines.markersize'] = 10



#Training Fata:
se=1
np.random.seed(se)
dataframe = pandas.read_csv("3fgl_associated_AGNandPSR_final_newindices_withclasses2_all.csv", header=None)
dataset1 = dataframe.values 
np.random.shuffle(dataset1[1:])
X = dataset1[1:,0:10].astype(float)
print(dataset1[0])
#Y = dataset[1:1933,5]
Y = dataset1[1:,11]
#dataset=dataset[:,:5]
num=[]
num2=[]
resul=[]

weight1=800/166
weight2=800/1739
class_weights= {'PSR': weight1, 'AGN': weight2}
# encode class values as integers

encoder = preprocessing.LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

#Testing Data, although we aren't using it at the moment since we are more concerned with the classfication domains of the training data:
dataframe = pandas.read_csv("3fgl_unass_withclasses_nn_allfeat.csv", header=None)
dataset = dataframe.values
X2 = dataset[1:,3:9].astype(float)
Y2 = dataset[1:,11]
encoder = preprocessing.LabelEncoder()
encoder.fit(Y2)
Y2 = encoder.transform(Y2)



h = .02  # step size in the mesh



#Much of the code was taken from elsewhere so names and classifiers function below, but we don't need it at the moment
names = ["Random Forest", "Neural Net", "AdaBoost"]

classifiers = [RandomForestClassifier(max_depth=200, n_estimators=12, class_weight='balanced',oob_score=True),
    MLPClassifier(alpha=0.001, max_iter=200),
    AdaBoostClassifier()]




#X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
 #                         random_state=1, n_clusters_per_class=1)

#Making datsets, these were initially with different features and not the same:
X1=dataset1[1:,2:4]
print(X1)
y1=Y

X2=dataset1[1:,2:4]
y2=Y

X3=dataset1[1:,2:4]
y3=Y


#rng = np.random.RandomState(2)
#X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X1, y1)
linearly_separable2 = (X2, y2)
linearly_separable3 = (X3, y3)
print(linearly_separable)
datasets = [#make_moons(noise=0.3, random_state=0),
            #make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable2,linearly_separable,linearly_separable3
            ]




X, y = datasets[1]
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
ax1.set_ylim((-2,5))
ax1.set_xlabel('Spectral Index')
ax1.set_ylabel('Log Significant Curvature')
ax1.set_title('Input Data')
#ax1.set_xticks(())
#ax1.set_yticks(())
plt.show()

#Main part of code:

fig1,ax2 = plt.subplots()

#Choose classifier:
clf=RandomForestClassifier(max_depth=6, n_estimators=50, class_weight='balanced',oob_score=True)
#clf= MLPClassifier(max_iter=50,hidden_layer_sizes=(10,), activation='tanh', solver='adam',batch_size=1000).fit(X_train,y_train)
#clf=GradientBoostingClassifier(n_estimators=100, learning_rate=0.3,max_depth=3, random_state=0).fit(X_train,y_train)
#clf=LogisticRegression(max_iter=300, C=0.1,solver='lbfgs').fit(X_train,y_train)
clf.fit(X_train, y_train)

lenth=len(X_test)
score = clf.score(X_test, y_test)       #Score of our classifier
i=0


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
if hasattr(clf, "decision_function"):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
print(y_test_psr)
        # Put the result into a color plot
Z = Z.reshape(xx.shape)
cs=ax2.contourf(xx, yy, Z, cmap='binary', alpha=.8)
fig1.colorbar(cs)
        # Plot the training points
scatter1=ax2.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='tab20b',edgecolors='k',marker='x')
        # Plot the testing points

scatter2=ax2.scatter(X_test_spec_agn, X_test_sig_agn, c=y_test_agn, cmap='tab20b', alpha=0.8,edgecolors='k')
scatter3=ax2.scatter(X_test_spec_psr, X_test_sig_psr, c=y_test_psr, cmap='cool', alpha=1,edgecolors='k')

#Make legends:
k=scatter1.legend_elements()[0]
k2=scatter2.legend_elements()[0]
k3=scatter3.legend_elements()[0]

k4=k+k2+k3
k4=np.asarray(k4)
legend1 = ax2.legend(k4,('Train AGN', 'Train PSR','Test AGN','Test PSR'),loc='upper right',fontsize=16)

#print(clf.n_layers_)
ax2.set_xlim(xx.min(), xx.max())
ax2.set_title('Random Forest (50,6)')
#ax2.set_ylim(yy.min(), yy.max())
ax2.set_xlabel('Spectral Index')
ax2.set_ylabel('Log(Significant_Curvature)')
ax2.set_ylim((-2,5))
#ax.set_xticks(np.arange(-5,3,step=1))
 #       ax.set_yticks(())
  #      if ds_cnt == 0:
   #         ax.set_title(name)
ax2.text(xx.max() , yy.max() + .3, ('%.2f' % score).lstrip('0'),
               size=15, horizontalalignment='right')
 #       i += 1

'''        
figure = plt.figure(figsize=(27, 9))
figure.suptitle("Classification Domains of Different Algorithms")
i = 1
print(datasets[0])
# iterate over datasets
for ds_cnt, ds in enumerate(datasets[1:2]):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler(with_mean=False).fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.3, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    #ax.legend(('Red=AGN', 'Blue=PSR'))
    
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
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(np.arange(-5,3,step=1))
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() , yy.max() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1
'''
plt.tight_layout()
plt.show()
