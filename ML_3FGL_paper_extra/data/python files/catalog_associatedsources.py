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

pyplot.rcParams['xtick.labelsize'] = 16
pyplot.rcParams['axes.labelsize'] = 16
pyplot.rcParams['axes.titlesize'] = 25
pyplot.rcParams['font.size'] = 15
pyplot.rcParams['ytick.labelsize'] = 16


se=0
valscore3=0
#pro1=np.zeros((1905,9))
pro1source=[]
dataframe = pandas.read_csv("./files/3fgl_assoc_newfeats.csv", header=None)
#dataframe = pandas.read_csv("4fgl_assoc_3.csv", header=None)

dataset1 = dataframe.values

rf=[]
lr=[]
bdt=[]
nn=[]

lenth=2408


while se<1000:
    #data:
    np.random.seed(se)
    
    dataframe2 = pandas.read_csv("./files/3fgl_4fgl_newfeats.csv", header=None)
    dataset2 = dataframe2.values
    
    np.random.shuffle(dataset1[1:])


    X=[dataset1[i,1:12].astype(float) for i in range(len(dataset1)) if dataset1[i,12]=='AGN' or dataset1[i,12]=='PSR']
    Y =[dataset1[i,12] for i in range(len(dataset1)) if dataset1[i,12]=='AGN' or dataset1[i,12]=='PSR']
    encoder = preprocessing.LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)
    train1,val_inp1, train_truth1,  val_out1 = train_test_split(X, Y, test_size=.3, random_state=se)       #Split into training and validation

    val_out1=np.ravel(val_out1)                     #ravel is used since flattened label array required
    train_truth1=np.ravel(train_truth1)
    
    count=0
    #pro2=pro1
    oversample = RandomOverSampler(sampling_strategy='minority')
    X_over, y_over = oversample.fit_resample(train1, train_truth1)
    #X_over, y_over=train1,train_truth1


    X2=[dataset2[i,1:12].astype(float) for i in range(len(dataset2)) if dataset2[i,12]=='AGN' or dataset2[i,12]=='PSR']
    Y2 =[dataset2[i,12] for i in range(len(dataset2)) if dataset2[i,12]=='AGN' or dataset2[i,12]=='PSR']
    encoder = preprocessing.LabelEncoder()
    encoder.fit(Y2)
    Y2 = encoder.transform(Y2)
    testdatainput=X2
    testdatalabels=np.ravel(Y2)                     #ravel is used since flattened label array required

    clf= GradientBoostingClassifier(n_estimators=100, learning_rate=0.3,max_depth=2).fit(X_over, y_over)
    clf2= MLPClassifier(max_iter=300,hidden_layer_sizes=(11,), activation='tanh', solver='lbfgs').fit(X_over, y_over)
    clf3= LogisticRegression(max_iter=200, C=2,solver='lbfgs').fit(X_over, y_over)
    clf4 = RandomForestClassifier(n_estimators=50,max_depth=6,oob_score=True)
    clf4.fit(X_over, y_over)
    
  
    fit1=clf.score(testdatainput,testdatalabels)
    fit2=clf2.score(testdatainput,testdatalabels)
    fit3=clf3.score(testdatainput,testdatalabels)
    fit4=clf4.score(testdatainput,testdatalabels)
    
    rf.append(fit4)
    lr.append(fit3)
    nn.append(fit2)
    bdt.append(fit1)
    se=se+1
    print(se)


#prop1=prop1/1000
print('means(rf,lr,nn,bdt):',np.mean(rf),np.mean(lr),np.mean(nn),np.mean(bdt))
print('std:',np.std(rf),np.std(lr),np.std(nn),np.std(bdt))
