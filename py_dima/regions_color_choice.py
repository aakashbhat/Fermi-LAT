# python regions_color_choice.py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import plotting

h = .02  # step size in the mesh

names = ["Nearest Neighbors", #"Linear SVM", "RBF SVM", "Gaussian Process",
         #"Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         #"Naive Bayes", "QDA"
         ]

classifiers = [
    KNeighborsClassifier(3),
    #SVC(kernel="linear", C=0.025),
    #SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    #DecisionTreeClassifier(max_depth=5),
    #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #MLPClassifier(alpha=1, max_iter=1000),
    #AdaBoostClassifier(),
    #GaussianNB(),
    #QuadraticDiscriminantAnalysis()
    ]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)


datasets = [make_moons(noise=0.3, random_state=0, n_samples=(300, 3000)),
            #make_circles(noise=0.2, factor=0.5, random_state=1),
            #linearly_separable
            ]

plotting.setup_figure_pars()

figure = plt.figure(figsize=(9, 6))
i = 1
levels = np.arange(0., 1.01, 0.1)
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)
    
    c1_train_inds = [i for i in range(len(y_train)) if y_train[i] < 0.5]
    c1_test_inds = [i for i in range(len(y_test)) if y_test[i] < 0.5]
    c2_train_inds = [i for i in range(len(y_train)) if y_train[i] >= 0.5]
    c2_test_inds = [i for i in range(len(y_test)) if y_test[i] >= 0.5]

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    #cm = plt.cm.RdBu
    cm = plt.cm.GnBu_r
    norm = colors.Normalize(vmin=0, vmax=0.9)
    
    #cm_bright = ListedColormap(['#00FF00', '#0000FF'])
    trainc1_color = 'green'
    trainc1_marker = 's'
    testc1_color = 'yellow'
    testc1_marker = 'd'
    
    trainc2_color = 'blue'
    trainc2_marker = '^'
    testc2_color = 'magenta'
    testc2_marker = 'v'
    
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers), i)
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
        cs = ax.contourf(xx, yy, Z, cmap=cm, alpha=0.7, levels=levels)
        #ax.imshow(xx, yy, Z, cmap=cm, alpha=.8)
        
        alpha = 0.8
        # Training points for class 2
        ax.scatter(X_train[c2_train_inds, 0], X_train[c2_train_inds, 1], c=trainc2_color, alpha=alpha,
                   marker=trainc2_marker, edgecolors='k', label='AGN training')
        # Testing points for class 2
        ax.scatter(X_test[c2_test_inds, 0], X_test[c2_test_inds, 1], c=testc2_color, alpha=alpha,
                   marker=testc2_marker, edgecolors='k', label='AGN testing')
        # Training points for class 1
        ax.scatter(X_train[c1_train_inds, 0], X_train[c1_train_inds, 1], c=trainc1_color, alpha=alpha,
                   marker=trainc1_marker, edgecolors='k', label='PSR training')
        # Testing points for class 1
        ax.scatter(X_test[c1_test_inds, 0], X_test[c1_test_inds, 1], c=testc1_color, alpha=alpha,
                   marker=testc1_marker, edgecolors='k', label='PSR testing')

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1
        ax.legend()
        cbar = figure.colorbar(cs, ax=ax, shrink=0.9)
        plt.xlabel('Spectral Index')
        plt.ylabel('Ln(Significant Curvature)')

#plt.tight_layout()
#plt.show()
if not os.path.isdir('plots/'):
    os.mkdir('plots/')
plt.savefig('plots/choose_domain_color.pdf')
