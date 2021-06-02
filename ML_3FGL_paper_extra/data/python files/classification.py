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
from imblearn.over_sampling import RandomOverSampler
import imblearn as imbl
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from numpy import where
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
#plt.rcParams['lines.markersize'] = 8

score=0
#Training Fata:
se=0
s1=11339
s2=652
zbig=np.zeros((s1,s2))
while se<10:
    np.random.seed(se)
    dataframe = pandas.read_csv("./files/4fgldr2_all_newfeats.csv", header=None)
    dataset1 = dataframe.values 
    np.random.shuffle(dataset1[1:])


    X=[dataset1[i,[10,9]].astype(float) for i in range(len(dataset1)) if dataset1[i,17]=='AGN' or dataset1[i,17]=='PSR']#or dataset1[i,12]=='OTHER']
    Y =[dataset1[i,17] for i in range(len(dataset1)) if dataset1[i,17]=='AGN' or dataset1[i,17]=='PSR']#or dataset1[i,12]=='OTHER']
    encoder = preprocessing.LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)
    counter=Counter(Y)
    X=np.asarray(X)
    #print(X)
    h = .02  # step size in the mesh
    


    #X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
     #                         random_state=1, n_clusters_per_class=1)

    #Making datsets, these were initially with different features and not the same:
    X1=X
    #print(dataset1[0,2:4])
    y=Y
    linearly_separable = (X1, y)
    #print(linearly_separable)
    datasets = [linearly_separable]
    X, y = datasets[0]



        #X = StandardScaler(with_mean=False,with_std=False).fit_transform(X)
    X_t, X_test, y_t, y_test = \
        train_test_split(X1, y, test_size=.3, random_state=0)       #Split into training and validation

    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_t, y_t)
    print(se)
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5   #Define minimumm and maximum of axes
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    #cm = plt.cm.RdBu
    #cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    #ax.legend(('Red=AGN', 'Blue=PSR'))
    levels = np.arange(0., 1.01, 0.1)
    cm = plt.cm.RdYlBu
    norm = colors.Normalize(vmin=0, vmax=1.0)
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
    #Main part of code:
    #c1_train_inds = [i for i in range(len(y_train)) if y_train[i] ==2]
    #c1_test_inds = [i for i in range(len(y_test)) if y_test[i] ==2]
    c2_train_inds = [i for i in range(len(y_train)) if y_train[i] == 0]
    c2_test_inds = [i for i in range(len(y_test)) if y_test[i] == 0]
    c3_train_inds = [i for i in range(len(y_train)) if y_train[i] == 1]
    c3_test_inds = [i for i in range(len(y_test)) if y_test[i] == 1]

    trainc3_color = 'green'
    trainc3_marker = 's'
# for some reason squares look visually larger than triangles
# this is a hack to make the a bit smaller
    trainc1_marker_size = plt.rcParams['lines.markersize'] - 0.5 

    testc3_color = 'yellow'
    testc3_marker = 'd'
    
    trainc2_color = 'blue'
    trainc2_marker = '^'
    testc2_color = 'magenta'
    testc2_marker = 'v'
    #trainc1_color = 'red'
    #trainc1_marker = 'x'
    #testc1_color = 'orange'
    #testc1_marker = 'D'




#Choose classifier:
    #clf=RandomForestClassifier(max_depth=6, n_estimators=50,oob_score=True)
    #clf= MLPClassifier(max_iter=600,hidden_layer_sizes=(2,), activation='tanh', solver='lbfgs').fit(X_train,y_train)
    #clf=GradientBoostingClassifier(n_estimators=20, learning_rate=0.3,max_depth=2, random_state=0).fit(X_train,y_train)
    clf=LogisticRegression(max_iter=200, C=0.1,solver='lbfgs').fit(X_train,y_train)
    #clf.fit(X_train, y_train)

    lenth=len(X_test)
    score = score+clf.score(X_test, y_test)       #Score of our classifier
    i=0
    levels = np.arange(0., 1.01, 0.1)
    cm = plt.cm.GnBu_r
    norm = colors.Normalize(vmin=0, vmax=0.9)
    #cm=plt.cm.rainbow
    #norm = colors.Normalize(vmin=-1, vmax=3)

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
    #Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#else:
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
# Put the result into a color plot
    Z = Z.reshape(xx.shape)
    zbig=zbig+Z
    se=se+1
fig1,ax2 = plt.subplots()

zbig=(zbig/10)#.astype(int)
cs=ax2.contourf(xx, yy, zbig, cmap=cm,norm=norm, alpha=.8,levels=levels)
fig1.colorbar(cs,ax=ax2,shrink=0.9)
        # Plot the training points
score=score/10
alpha=0.9
k=1
'''
for i in range(len(X_train)):
    l=X_train[i]
    j=i+1
    while(j<len(X_train)):
        m=X_train[j]
        if l[0]==m[0]and l[1]==m[1]:
            X_train[j]=X_train[j]+k*2/90
            k=k+1
        j=j+1
    k=1
'''
'''
alphas = np.linspace(1, 0.1, 1221)
rgba_colors = np.zeros((1221,4))
# for red the first column needs to be one
rgba_colors[:,1] =1,
# the fourth column needs to be your alphas
rgba_colors[:, 3] = alphas
'''
# Training points for class 2
ax2.scatter(X_train[c2_train_inds, 0], X_train[c2_train_inds, 1], c=trainc2_color, alpha=alpha,
                   marker=trainc2_marker, edgecolors='k', label='AGN training')
        # Testing points for class 2
ax2.scatter(X_test[c2_test_inds, 0], X_test[c2_test_inds, 1], c=testc2_color, alpha=alpha,
                   marker=testc2_marker, edgecolors='k', label='AGN testing')
        # Training points for class 1
#ax2.scatter(X_train[c1_train_inds, 0], X_train[c1_train_inds, 1], color=trainc1_color,
 #                  marker=trainc1_marker, edgecolors='k', label='PSR training')
        # Testing points for class 1
#ax2.scatter(X_test[c1_test_inds, 0], X_test[c1_test_inds, 1], c=testc1_color, alpha=alpha,
#                   marker=testc1_marker, edgecolors='k', label='PSR testing')
ax2.scatter(X_train[c3_train_inds, 0], X_train[c3_train_inds, 1], color=trainc3_color,
                   marker=trainc3_marker, edgecolors='k', label='PSR training')
        # Testing points for class 1
ax2.scatter(X_test[c3_test_inds, 0], X_test[c3_test_inds, 1], c=testc3_color, alpha=alpha,
                   marker=testc3_marker, edgecolors='k', label='PSR testing')

ax2.legend(loc=1)
ax2.text(6.5,0.3,"Iterations: 200")
ax2.text(6.5,0,"Solver: LBFGS")
#ax2.text(0.02,-1.3,"Trees: 100")
#ax2.text(0.02,-1.6,"Maximum Depth: 6")
#ax2.set_xlim(xx.min(), xx.max())
ax2.set_title('Logistic Regression')
#ax2.set_title('Random Forests (3-Class)')

#ax2.set_ylim(yy.min(), yy.max())
ax2.set_xlabel('Ln(Variability_Index)')
ax2.set_ylabel('Signif_Curv')
ax2.set_ylim((-0.5,15))
ax2.set_xlim((0,8))

#ax.set_xticks(np.arange(-5,3,step=1))
 #       ax.set_yticks(())
  #      if ds_cnt == 0:
   #         ax.set_title(name)
ax2.text(6.5 , -0.3, ('Testing Score:%.2f' % score).lstrip('0'))
 #       i += 1


#plt.tight_layout()
plt.show()
fn = 'plots/rf_50_6_3class.pdf'
print('save plot to file')
print(fn)
#plt.savefig(fn)

