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
from imblearn.datasets import make_imbalance
from sklearn.preprocessing import StandardScaler


pyplot.rcParams['xtick.labelsize'] = 16
pyplot.rcParams['axes.labelsize'] = 16
pyplot.rcParams['axes.titlesize'] = 25
pyplot.rcParams['font.size'] = 15
pyplot.rcParams['ytick.labelsize'] = 16


se=0
valscore3=0
pro1source=[]
#dataframe = pandas.read_csv("./files/4fgldr2_all_newfeats.csv", header=None)
dataframe = pandas.read_csv("./files/3fgl_all_newfeats.csv", header=None)

dataset1 = dataframe.values
k=dataset1[1:,1].astype(float)
dataset1[1:,1]=np.cos(k)
rf=[]
lr=[]
bdt=[]
nn=[]
rfo=[]
lro=[]
bdto=[]
nno=[]
lenth=12


while se<1000:
    #data:
    np.random.seed(se)
    dataframe2 = pandas.read_csv("./files/3fgl_4fgl_newfeats_3class.csv", header=None)
    dataset2 = dataframe2.values
    l=dataset2[1:,1].astype(float)
    dataset2[1:,1]=np.cos(l)
    np.random.shuffle(dataset1[1:])


    X=[dataset1[i,1:lenth].astype(float) for i in range(len(dataset1)) if dataset1[i,lenth]=='AGN' or dataset1[i,lenth]=='PSR' or dataset1[i,lenth]=='OTHER']
    Y =[dataset1[i,lenth] for i in range(len(dataset1)) if dataset1[i,lenth]=='AGN' or dataset1[i,lenth]=='PSR'or dataset1[i,lenth]=='OTHER']
    print(Y)
    encoder = preprocessing.LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)
    #X = StandardScaler(with_mean=False,with_std=False).fit_transform(X)

    train1,val_inp1, train_truth1,  val_out1 = train_test_split(X, Y, test_size=.3, random_state=se)       #Split into training and validation
    val_out1=np.ravel(val_out1)                     #ravel is used since flattened label array required
    train_truth1=np.ravel(train_truth1)

    
    #################################
    Y0=[i for i in range(len(train_truth1)) if train_truth1[i]==0]
    Y1=[i for i in range(len(train_truth1)) if train_truth1[i]==1]
    Y2=[i for i in range(len(train_truth1)) if train_truth1[i]==2]
    w1=int(np.sqrt(len(Y0)*len(Y1)))
    w2=int(np.sqrt(len(Y0)*len(Y2)))
    weight={0:len(Y0),1:w1,2:w2}
    print('OTHER;PSR:',len(Y1),len(Y2))

    #################################

    
    X2=[dataset2[i,1:lenth].astype(float) for i in range(len(dataset2)) if dataset2[i,lenth]=='AGN' or dataset2[i,lenth]=='PSR'or dataset1[i,12]=='OTHER']
    Y2 =[dataset2[i,lenth] for i in range(len(dataset2)) if dataset2[i,lenth]=='AGN' or dataset2[i,lenth]=='PSR'or dataset1[i,12]=='OTHER']
    encoder = preprocessing.LabelEncoder()
    encoder.fit(Y2)
    Y2 = encoder.transform(Y2)
    
    
    testdatainput=X2
    testdatalabels=Y2                     #ravel is used since flattened label array required
    
    ######################################
    count=0
    
    clf5= GradientBoostingClassifier(n_estimators=100, learning_rate=0.3,max_depth=2).fit(train1, train_truth1)
    clf6= MLPClassifier(max_iter=300,hidden_layer_sizes=(11,), activation='tanh', solver='lbfgs').fit(train1, train_truth1)
    clf7= LogisticRegression(max_iter=200, C=2,solver='lbfgs').fit(train1, train_truth1)
    clf8 = RandomForestClassifier(n_estimators=50,max_depth=6,oob_score=True)
    clf8.fit(train1, train_truth1)
    
  
    fit5=clf5.score(testdatainput,testdatalabels)
    fit6=clf6.score(testdatainput,testdatalabels)
    fit7=clf7.score(testdatainput,testdatalabels)
    fit8=clf8.score(testdatainput,testdatalabels)
    
    rf.append(fit8)
    lr.append(fit7)
    nn.append(fit6)
    bdt.append(fit5)
    
    ##########################################

    oversample = RandomOverSampler(sampling_strategy=weight)
    X_over, y_over = oversample.fit_resample(train1, train_truth1)

 
    clf= GradientBoostingClassifier(n_estimators=100, learning_rate=0.3,max_depth=2).fit(X_over, y_over)
    clf2= MLPClassifier(max_iter=600,hidden_layer_sizes=(11,), activation='tanh', solver='lbfgs').fit(X_over, y_over)
    clf3= LogisticRegression(max_iter=500, C=1,solver='lbfgs').fit(X_over, y_over)
    clf4 = RandomForestClassifier(n_estimators=50,max_depth=6,oob_score=True)
    clf4.fit(X_over, y_over)
    
  
    fit1=clf.score(testdatainput,testdatalabels)
    fit2=clf2.score(testdatainput,testdatalabels)
    fit3=clf3.score(testdatainput,testdatalabels)
    fit4=clf4.score(testdatainput,testdatalabels)
    '''
    fit5=clf.score(testdatainput,testdatalabels)
    fit6=clf2.score(testdatainput,testdatalabels)
    fit7=clf3.score(testdatainput,testdatalabels)
    fit8=clf4.score(testdatainput,testdatalabels)
    
    rf.append(fit8)
    lr.append(fit7)
    nn.append(fit6)
    bdt.append(fit5)
    '''
    rfo.append(fit4)
    lro.append(fit3)
    nno.append(fit2)
    bdto.append(fit1)
    se=se+1
    print(se)
    

#prop1=prop1/1000
print('means4fgl(rf,lr,bdt.nn):',np.mean(rf),np.mean(lr),np.mean(bdt),np.mean(nn))
print('std:',np.std(rf),np.std(lr),np.std(bdt),np.std(nn))
print('means4fgloversampled(rf,lr,nn,bdt):',np.mean(rfo),np.mean(lro),np.mean(bdto),np.mean(nno))
print('stdo:',np.std(rfo),np.std(lro),np.std(nno),np.std(bdto),np.std(nno))


