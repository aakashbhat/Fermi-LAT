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

pyplot.rcParams['xtick.labelsize'] = 16
pyplot.rcParams['axes.labelsize'] = 16
pyplot.rcParams['axes.titlesize'] = 25
pyplot.rcParams['font.size'] = 15
pyplot.rcParams['ytick.labelsize'] = 16


se=0
valscore3=0
pro1=np.zeros((1008,8))
feat2=np.zeros(17)
prop1=np.zeros((1008,4))


while se<10:
    #data:
    np.random.seed(se)
    #dataframe = pandas.read_csv("4fgl_assoc_3.csv", header=None)
    #dataframe = pandas.read_csv("./files/4fgl_assoc.csv", header=None)
    dataframe = pandas.read_csv("./files/3fgl_associated_AGNandPSR.csv", header=None)

    dataset1 = dataframe.values 
    np.random.shuffle(dataset1[1:])
    X = dataset1[1:,0:10].astype(float)
    #print(dataset1[0:,2:16])
    #print(dataset1[0,:])
    Y = dataset1[1:,11]
    print(Y)
    '''
    weight1=800/166
    weight2=800/1739
    class_weights= {'PSR': weight1, 'AGN': weight2}
    # encode class values as integers

    '''
    encoder = preprocessing.LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)
    print(Y)
    #dataframe = pandas.read_csv("4fgl_unassoc_3_2.csv", header=None)
    #dataframe = pandas.read_csv("3fgl_allunassoc.csv", header=None)
    #dataframe = pandas.read_csv("3fgl_assoc_notagnpsr.csv", header=None)
    #dataframe = pandas.read_csv("4fgl_others_3.csv", header=None)
    '''
    dataframe = pandas.read_csv("./catas/3FGL_unassocvs4FGLassoc_AGN&PSR_catalog_unweighted.csv", header=None)
    dataset = dataframe.values 

    X2 = dataset[1:,1:11].astype(float)
    Y2 = dataset[1:,44]
    print(dataset[0,44])
    encoder = preprocessing.LabelEncoder()
    encoder.fit(Y2)
    Y2j = encoder.transform(Y2)
    '''
    #print(Y2)
    train1,val_inp1, train_truth1,  val_out1 = train_test_split(X, Y, test_size=.3, random_state=se)       #Split into training and validation
    '''
    train1=X[0:]                    
    train_truth1=Y[0:]
    val_inp1=X2[0:]
    val_out1=Y2j[0:]
    '''
    val_out1=np.ravel(val_out1)                     #ravel is used since flattened label array required
    train_truth1=np.ravel(train_truth1)
    valscore2=valscore3

    #prop2=prop1
    #pro2=pro1
    #clf4= GradientBoostingClassifier(n_estimators=100, learning_rate=0.3,max_depth=2).fit(train1, train_truth1)
    clf4= MLPClassifier(max_iter=500,hidden_layer_sizes=(10,10,), activation='tanh', solver='adam',early_stopping=False).fit(train1,train_truth1)
    #clf4= LogisticRegression(max_iter=200, C=2,solver='lbfgs').fit(train1, train_truth1)
    #clf4 = RandomForestClassifier(n_estimators=100,max_depth=12,oob_score=True)
    #clf4.fit(train1,train_truth1)
    valscore3=clf4.score(val_inp1,val_out1)
    '''
    pro=np.zeros((1008,8))
    pro[:,0:2]=clf.predict_proba(val_inp1)
    pro[:,2:4]=clf2.predict_proba(val_inp1)
    pro[:,4:6]=clf3.predict_proba(val_inp1)
    pro[:,6:8]=clf4.predict_proba(val_inp1)

    #pro=clf.predict_proba(val_inp1)
    pro1=(pro2+pro)

    prop=np.zeros((1008,4))
    prop[:,0]=clf.predict(val_inp1)
    prop[:,1]=clf2.predict(val_inp1)
    prop[:,2]=clf3.predict(val_inp1)
    prop[:,3]=clf4.predict(val_inp1)
    
    '''
    valscore3=(valscore3+valscore2)
    se=se+1
    #print(valscore3)
    #pro=clf.predict_proba(val_inp1)
    #prop1=(prop2+prop)
    print(se)
    #feat=clf.feature_importances_
    #feat2=feat2+feat


#pro1=pro1/1000
#prop1=prop1/1000
#print(pro1)
#result=np.hstack((dataset[1:],pro1))
#result2=np.hstack((result,prop1))
#print(result2)

#print(clf.feature_importances_)
#print(result)
print(valscore3/10)
#print(feat2/1000)
#result=pandas.DataFrame(result)
#result2=pandas.DataFrame(result2)
#result2.to_csv(path_or_buf="3fgl_unassoc_catalog_unweighted.csv",index=False)
    
