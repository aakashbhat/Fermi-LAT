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
import math

pyplot.rcParams['xtick.labelsize'] = 16
pyplot.rcParams['axes.labelsize'] = 16
pyplot.rcParams['axes.titlesize'] = 25
pyplot.rcParams['font.size'] = 15
pyplot.rcParams['ytick.labelsize'] = 16

z1=287
z2=1336
z3=2027

#z3=357
se=0
valscore3=0
pro1=np.zeros((z3,16))
feat2=np.zeros(17)
prop1=np.zeros((z3,4))


while se<1000:
    #data:
    np.random.seed(se)
    dataframe = pandas.read_csv("./files/4fgldr2_assoc.csv", header=None)
    #dataframe = pandas.read_csv("./files/3fgl_associated_AGNandPSR.csv", header=None)

    dataset1 = dataframe.values 
    np.random.shuffle(dataset1[1:])
    X = dataset1[1:,0:16].astype(float)
    #print(dataset1[0,:])
    Y = dataset1[1:,16]
    
    encoder = preprocessing.LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)





    dataframe = pandas.read_csv("./files/4fgldr2_nonassoc.csv", header=None)
    #dataframe = pandas.read_csv("./files/3fgl_allunassoc.csv", header=None)
    #dataframe = pandas.read_csv("./files/3fgl_assoc_notagnpsr.csv", header=None)
    #dataframe = pandas.read_csv("./files/4fgldr2_other.csv", header=None)
    #dataframe = pandas.read_csv("3fgl_unassoc_4fgl_assoc.csv", header=None)

    dataset = dataframe.values
    print(Y)
    X2 = dataset[1:,1:17].astype(float)
    X2prob=dataset[1:,22:38].astype(float)
    print(X2prob)
    train1=X[0:]                    
    train_truth1=Y[0:]
    val_inp1=X2[0:]
    probs=X2prob[0:]
    #val_out1=Y[2400:]
    
    #val_out1=np.ravel(val_out1)                     #ravel is used since flattened label array required
    train_truth1=np.ravel(train_truth1)
    valscore2=valscore3

    pro2=pro1

    X_over, y_over=train1, train_truth1
    clf= GradientBoostingClassifier(n_estimators=100, learning_rate=0.3,max_depth=2).fit(X_over, y_over)
    clf2= MLPClassifier(max_iter=300,hidden_layer_sizes=(16,), activation='tanh', solver='adam').fit(X_over, y_over)
    clf3= LogisticRegression(max_iter=200, C=2,solver='lbfgs').fit(X_over, y_over)
    clf4 = RandomForestClassifier(n_estimators=50,max_depth=6,oob_score=True)
    clf4.fit(X_over, y_over)
     

    oversample = RandomOverSampler(sampling_strategy='minority')
    X_over0, y_over0 = oversample.fit_resample(train1, train_truth1)
    clf5= GradientBoostingClassifier(n_estimators=100, learning_rate=0.3,max_depth=2).fit(X_over0, y_over0)
    clf6= MLPClassifier(max_iter=300,hidden_layer_sizes=(16,), activation='tanh', solver='adam').fit(X_over0, y_over0)
    clf7= LogisticRegression(max_iter=200, C=2,solver='lbfgs').fit(X_over0, y_over0)
    clf8 = RandomForestClassifier(n_estimators=50,max_depth=6,oob_score=True)
    clf8.fit(X_over0, y_over0)
    
    

    pro=np.zeros((z3,16))
    pro[:,0:2]=pow((probs[:,0:2]-clf.predict_proba(val_inp1)),2)
    pro[:,2:4]=pow((probs[:,2:4]-clf2.predict_proba(val_inp1)),2)
    pro[:,4:6]=pow((probs[:,4:6]-clf3.predict_proba(val_inp1)),2)
    pro[:,6:8]=pow((probs[:,6:8]-clf4.predict_proba(val_inp1)),2)
    pro[:,8:10]=pow((probs[:,8:10]-clf5.predict_proba(val_inp1)),2)
    pro[:,10:12]=pow((probs[:,10:12]-clf6.predict_proba(val_inp1)),2)
    pro[:,12:14]=pow((probs[:,12:14]-clf7.predict_proba(val_inp1)),2)
    pro[:,14:16]=pow((probs[:,14:16]-clf8.predict_proba(val_inp1)),2)

    #pro=clf.predict_proba(val_inp1)
    pro1=(pro2+pro)

    se=se+1
    print(pro[:,0:2])
    print(se)




pro1=np.sqrt(pro1/1000)
#prop1=prop1/1000
#print(pro1)
#pro2=np.array((109,8))
pro2=["AGN_BDT","PSR_BDT","AGN_NN","PSR_NN","AGN_LR","PSR_LR","AGN_RF","PSR_RF","AGN_BDT_O","PSR_BDT_O","AGN_NN_O","PSR_NN_O","AGN_LR_O","PSR_LR_O","AGN_RF_O","PSR_RF_O"]
pro3=np.vstack((pro2,pro1))

print(pro3)
#prop2=["BDT_P","NN_P","LR_P","RF_P"]
#prop3=np.vstack((prop2,prop1))

result=np.hstack((dataset[0:],pro3))
#result2=np.hstack((result,prop3))
#print(result2)

#print(clf.feature_importances_)
#print(result)
#print(valscore3/1000)
#print(feat2/1000)
#result=pandas.DataFrame(result)
result=pandas.DataFrame(result)
result.to_csv(path_or_buf="./catas/4fgldr2_nonassoc_std.csv",index=False)
    

