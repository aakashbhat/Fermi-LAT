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

z1=287
z2=1336
z3=1670
#z3=357
se=0
valscore3=0
pro1=np.zeros((z3,8))
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
    print(len(Y))
    '''
    weight1=800/166
    weight2=800/1739
    class_weights= {'PSR': weight1, 'AGN': weight2}
    # encode class values as integers

    '''
    encoder = preprocessing.LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)
    dataframe = pandas.read_csv("./files/4fgldr2_unassoc.csv", header=None)
    #dataframe = pandas.read_csv("./files/3fgl_allunassoc.csv", header=None)
    #dataframe = pandas.read_csv("./files/3fgl_assoc_notagnpsr.csv", header=None)
    #dataframe = pandas.read_csv("./files/4fgldr2_other.csv", header=None)
    #dataframe = pandas.read_csv("3fgl_unassoc_4fgl_assoc.csv", header=None)

    dataset = dataframe.values
    print(Y)
    X2 = dataset[1:,0:16].astype(float)
    #Y2 = dataset[1:,10]
    #encoder = preprocessing.LabelEncoder()
    #encoder.fit(Y2)
    #Y2j = encoder.transform(Y2)
    #print(len(X2))

    train1=X[0:]                    
    train_truth1=Y[0:]
    val_inp1=X2[0:]
    #val_out1=Y[2400:]
    
    #val_out1=np.ravel(val_out1)                     #ravel is used since flattened label array required
    train_truth1=np.ravel(train_truth1)
    valscore2=valscore3

    prop2=prop1
    pro2=pro1
    #oversample = RandomOverSampler(sampling_strategy='minority')
    #X_over, y_over = oversample.fit_resample(train1, train_truth1)
    #oversample = RandomOverSampler(sampling_strategy=0.5)
    X_over, y_over=train1, train_truth1
    clf= GradientBoostingClassifier(n_estimators=100, learning_rate=0.3,max_depth=2).fit(X_over, y_over)
    clf2= MLPClassifier(max_iter=300,hidden_layer_sizes=(16,), activation='tanh', solver='adam').fit(X_over, y_over)
    clf3= LogisticRegression(max_iter=200, C=2,solver='lbfgs').fit(X_over, y_over)
    clf4 = RandomForestClassifier(n_estimators=50,max_depth=6,oob_score=True)
    clf4.fit(X_over, y_over)
    #valscore3=clf2.score(val_inp1,val_out1)
    
    
    
    pro=np.zeros((z3,8))
    pro[:,0:2]=clf.predict_proba(val_inp1)
    pro[:,2:4]=clf2.predict_proba(val_inp1)
    pro[:,4:6]=clf3.predict_proba(val_inp1)
    pro[:,6:8]=clf4.predict_proba(val_inp1)

    #pro=clf.predict_proba(val_inp1)
    pro1=(pro2+pro)
    
    prop=np.zeros((z3,4))
    prop[:,0]=clf.predict(val_inp1)
    prop[:,1]=clf2.predict(val_inp1)
    prop[:,2]=clf3.predict(val_inp1)
    prop[:,3]=clf4.predict(val_inp1)
    
    
    #valscore3=(valscore3+valscore2)
    se=se+1
    #print(valscore3)
    #pro=clf.predict_proba(val_inp1)
    prop1=(prop2+prop)
    print(se)
    #feat=clf.feature_importances_
    #feat2=feat2+feat




pro1=pro1/1000
prop1=prop1/1000
print(pro1)
#pro2=np.array((109,8))
pro2=["AGN_BDT","PSR_BDT","AGN_NN","PSR_NN","AGN_LR","PSR_LR","AGN_RF","PSR_RF"]
pro3=np.vstack((pro2,pro1))

print(pro3)
prop2=["BDT_P","NN_P","LR_P","RF_P"]
prop3=np.vstack((prop2,prop1))

result=np.hstack((dataset[0:],pro3))
result2=np.hstack((result,prop3))
print(result2)

#print(clf.feature_importances_)
#print(result)
#print(valscore3/1000)
#print(feat2/1000)
#result=pandas.DataFrame(result)
result2=pandas.DataFrame(result2)
result2.to_csv(path_or_buf="./catas/4fgldr2_assoc_unassoc.csv",index=False)
    

