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
from imblearn.over_sampling import RandomOverSampler


####
pyplot.rcParams['xtick.labelsize'] = 16
pyplot.rcParams['axes.labelsize'] = 16
pyplot.rcParams['axes.titlesize'] = 25
pyplot.rcParams['font.size'] = 15
pyplot.rcParams['ytick.labelsize'] = 16
#####



se = 0



pro1source = []
# dataframe = pandas.read_csv("./files/3fgl_associated_AGNandPSR.csv", header=None)
dataframe = pandas.read_csv("./files/4fgldr2_assoc.csv", header=None)

dataset2 = dataframe.values

sourcenames = dataset2[1:, 16]
print('sourcename:',sourcenames)

pro1 = np.array((3747, 10))


pro1 = dataset2[1:, 0:10]
pro1[:, 0] = sourcenames
pro1[:, 1:9] = []
pro1[:, 9] = 1

# dataframe = pandas.read_csv("./files/3fgl_associated_AGNandPSR.csv", header=None)
dataframe = pandas.read_csv("./files/4fgldr2_assoc.csv", header=None)
dataset1 = dataframe.values

print(pro1)
dataframe3 = pandas.read_csv("./files/4fgldr2_nonassoc.csv", header=None)
dataset3 = dataframe.values

##################

while se < 1:
    # data:
    np.random.seed(se)

    # dataframe = pandas.read_csv("4fgl_assoc_3.csv", header=None)
    #Training Data goes here:

    np.random.shuffle(dataset1[1:])
    X = dataset1[1:, 0:16].astype(float)
    Y = dataset1[1:, 16]
    encoder = preprocessing.LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)
    train1 = X[0:2622]
    train_truth1 = Y[0:2622]
    val_inp1 = X[2622:]
    val_source = dataset1[2623:, 16]
    val_out1 = Y[2622:]
    # print(val_source[1])
    val_out1 = np.ravel(val_out1)  # ravel is used since flattened label array required
    train_truth1 = np.ravel(train_truth1)
    X_over = train1
    y_over = train_truth1




    #oversample = RandomOverSampler(sampling_strategy='minority')
    #X_over, y_over = oversample.fit_resample(train1, train_truth1)
    # oversample = RandomOverSampler(sampling_strategy=0.5)






    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.3, max_depth=2).fit(X_over, y_over)
    clf2 = MLPClassifier(max_iter=300, hidden_layer_sizes=(16,), activation='tanh', solver='adam').fit(X_over, y_over)
    clf3 = LogisticRegression(max_iter=200, C=2, solver='lbfgs').fit(X_over, y_over)
    clf4 = RandomForestClassifier(n_estimators=50, max_depth=6, oob_score=True)
    clf4.fit(X_over, y_over)



    '''
    pro=np.zeros((1008,8))
    pro[:,0:2]=clf.predict_proba(val_inp1)
    pro[:,2:4]=clf2.predict_proba(val_inp1)
    pro[:,4:6]=clf3.predict_proba(val_inp1)
    pro[:,6:8]=clf4.predict_proba(val_inp1)
    '''
    fit1 = clf.predict_proba(val_inp1)
    fit2 = clf2.predict_proba(val_inp1)
    fit3 = clf3.predict_proba(val_inp1)
    fit4 = clf4.predict_proba(val_inp1)
    print(se)
    # rf1=clf4.score(val_inp1,val_out1)*100
    # gb1=clf.score(val_inp1,val_out1)*100
    # nn1=clf2.score(val_inp1,val_out1)*100
    # lr1=clf3.score(val_inp1,val_out1)*100

    # scorerf=scorerf+pow((rf1-rf),2)
    # scorenn=scorenn+pow((nn1-nn),2)
    # scoregb=scoregb+pow((gb1-gb),2)
    # scorelr=scorelr+pow((lr1-lr),2)
    # print(scorerf)
    for i in range(len(val_inp1)):
        for j in range(3747):
            if pro1[j, 0] == val_source[i]:
                pro1[j, 1:3] = pro1[j, 1:3] + fit1[i]
                pro1[j, 3:5] = pro1[j, 3:5] + fit2[i]
                pro1[j, 5:7] = pro1[j, 5:7] + fit3[i]
                pro1[j, 7:9] = pro1[j, 7:9] + fit4[i]
                pro1[j, 9] = pro1[j, 9] + 1

    se = se + 1
'''
scorerf=scorerf/1000
scoregb=scoregb/1000
scorelr=scorelr/1000
scorenn=scorenn/1000
'''

# print(scorerf,scoregb,scorenn,scorelr)

for i in range(3747):
    for j in range(8):
        if pro1[i, 9] != 1:
            pro1[i, j + 1] = pro1[i, j + 1] / (pro1[i, 9] - 1)
pro1[:, 9] = pro1[:, 9] - 1
# print(np.mean(pro1[:,9]))

# prop1=prop1/1000
# print(pro1)

# dataframe = pandas.read_csv("./files/3fgl_associated_AGNandPSR.csv", header=None)
dataframe = pandas.read_csv("./files/4fgldr2_assoc.csv", header=None)
dataset3 = dataframe.values
pro2 = ["Source_Name", "AGN_BDT", "PSR_BDT", "AGN_NN", "PSR_NN", "AGN_LR", "PSR_LR", "AGN_RF", "PSR_RF",
        "Times in Testing"]
pro3 = np.vstack((pro2, pro1))

result = np.hstack((dataset3[0:], pro3))

# print(clf.feature_importances_)
# print(result)
# print(valscore3/1000)
# print(feat2/1000)
# result=pandas.DataFrame(result)
result = pandas.DataFrame(result)
result.to_csv(path_or_buf="./catas/4fgldr2_assoc_catalog_o.csv", index=False)

