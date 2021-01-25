print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
import pandas
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import plotting_dima
plotting_dima.setup_figure_pars()
se=2
np.random.seed(se)

# import some data to play with
dataframe = pandas.read_csv("./files/3fgl_all_newfeats.csv", header=None)
dataset1 = dataframe.values 
np.random.shuffle(dataset1[1:])
dataset2=dataset1[1:]
X=[dataset1[i,[6,5]].astype(float) for i in range(len(dataset1)) if dataset1[i,12]=='AGN' or dataset1[i,12]=='PSR'or dataset1[i,12]=='OTHER']
Y =[dataset1[i,12] for i in range(len(dataset1)) if dataset1[i,12]=='AGN' or dataset1[i,12]=='PSR'or dataset1[i,12]=='OTHER']
#print(X)
h = .02  # step size in the mesh
encoder = preprocessing.LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
#X = StandardScaler(with_mean=False,with_std=False).fit_transform(X)

logreg = linear_model.LogisticRegression(C=1e5,solver='lbfgs',class_weight='balanced')
rf=RandomForestClassifier(max_depth=6, n_estimators=50,oob_score=True)
nn= MLPClassifier(max_iter=600,hidden_layer_sizes=(2,), activation='tanh', solver='lbfgs')

# we create an instance of Neighbours Classifier and fit the data.
nn.fit(X, Y)
print(X)
X1=np.array(X)
first=X1[:,0]
second=X1[:,1]
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = first.min() - .5, first.max() + .5
y_min, y_max = second.min() - .5, second.max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(first, second, c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Ln(Variability_Index)')
plt.ylabel('Ln(Significant_Curvature)')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
#plt.show()
fn = 'plots/nn_600_lbfgs_multi.pdf'
print('save plot to file')
print(fn)
plt.savefig(fn)
