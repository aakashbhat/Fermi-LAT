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

rf=[]
bdt=[]
nn=[]
lr=[]


dataframe = pandas.read_csv("./files/3fgl_missingvals.csv", header=None)
dataset1 = dataframe.values
lenth=12

unassocnames=dataset1[1:,0]


###################
probs_ass_bdt=[]
probs_un_bdt=[]
probs_ass_nn=[]
probs_un_nn=[]
probs_ass_rf=[]
probs_un_rf=[]
probs_ass_lr=[]
probs_un_lr=[]
#####################
times=1000

while se<times:
    #data:
    np.random.seed(se)
    dataframe3 = pandas.read_csv("./files/3fgl_all_newfeats.csv", header=None)
    dataset3 = dataframe3.values
    #k=dataset3[1:,1].astype(float)
    #dataset3[1:,1]=np.cos(k)
    X2 = dataset1[1:,1:lenth].astype(float) 
    np.random.shuffle(dataset3[1:])
    
    ##############################
    X = [dataset3[i,1:lenth].astype(float) for i in range(len(dataset3)) if dataset3[i,lenth]=='AGN' or dataset3[i,lenth]=='PSR']#or dataset3[i,lenth]=='OTHER']
    Y = [dataset3[i,lenth] for i in range(len(dataset3)) if dataset3[i,lenth]=='AGN' or dataset3[i,lenth]=='PSR']#or dataset3[i,lenth]=='OTHER']
    #val_source1=[dataset3[i,0] for i in range(len(dataset3)) if dataset3[i,lenth]=='AGN' or dataset3[i,lenth]=='PSR'or dataset3[i,lenth]=='OTHER']

    encoder = preprocessing.LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)
    ################################
    lenth2=len(Y)
    lenth2=int(0.7*lenth2)
    train1=X[0:lenth2]                    
    train_truth1=Y[0:lenth2]
    val_inp1=X[lenth2:]
    val_out1=Y[lenth2:]
    
    val_out1=np.ravel(val_out1)                     #ravel is used since flattened label array required
    train_truth1=np.ravel(train_truth1)


    ################
    Y0=[i for i in range(len(train_truth1)) if train_truth1[i]==0]
    Y1=[i for i in range(len(train_truth1)) if train_truth1[i]==1]
    Y2=[i for i in range(len(train_truth1)) if train_truth1[i]==2]
    w1=int(np.sqrt(len(Y0)*len(Y1)))
    w2=int(np.sqrt(len(Y0)*len(Y2)))
    weight={0:len(Y0),1:w1,2:w2}
    ##################


    #X_over=train1
    #y_over=train_truth1
    oversample = RandomOverSampler(sampling_strategy='minority')
    X_over, y_over = oversample.fit_resample(train1, train_truth1)
    print(len(y_over))
    #oversample = RandomOverSampler(sampling_strategy=0.5)
    clf= GradientBoostingClassifier(n_estimators=100, learning_rate=0.3,max_depth=2).fit(X_over, y_over)
    clf2= MLPClassifier(max_iter=300,hidden_layer_sizes=(11,), activation='tanh', solver='lbfgs').fit(X_over, y_over)
    clf3= LogisticRegression(max_iter=200, C=1,solver='lbfgs').fit(X_over, y_over)
    clf4 = RandomForestClassifier(n_estimators=50,max_depth=6,oob_score=True)
    clf4.fit(X_over, y_over)
    
    ##########################################
    
    probs_un_bdt.append(clf.predict_proba(X2))
    probs_un_nn.append(clf2.predict_proba(X2))
    probs_un_rf.append(clf4.predict_proba(X2))
    probs_un_lr.append(clf3.predict_proba(X2))
    ################################################
    
    ##########################################
    
    se=se+1
    print(se)

########################


####################
unassoc_names=unassocnames
AGN_BDT_mean=np.zeros(len(unassoc_names))
AGN_BDT_std=np.zeros(len(unassoc_names))
PSR_BDT_mean=np.zeros(len(unassoc_names))
PSR_BDT_std=np.zeros(len(unassoc_names))
OTHER_BDT_mean=np.zeros(len(unassoc_names))
OTHER_BDT_std=np.zeros(len(unassoc_names))
AGN_NN_mean=np.zeros(len(unassoc_names))
AGN_NN_std=np.zeros(len(unassoc_names))
PSR_NN_mean=np.zeros(len(unassoc_names))
PSR_NN_std=np.zeros(len(unassoc_names))
OTHER_NN_mean=np.zeros(len(unassoc_names))
OTHER_NN_std=np.zeros(len(unassoc_names))
AGN_RF_mean=np.zeros(len(unassoc_names))
AGN_RF_std=np.zeros(len(unassoc_names))
PSR_RF_mean=np.zeros(len(unassoc_names))
PSR_RF_std=np.zeros(len(unassoc_names))
OTHER_RF_mean=np.zeros(len(unassoc_names))
OTHER_RF_std=np.zeros(len(unassoc_names))
AGN_LR_mean=np.zeros(len(unassoc_names))
AGN_LR_std=np.zeros(len(unassoc_names))
PSR_LR_mean=np.zeros(len(unassoc_names))
PSR_LR_std=np.zeros(len(unassoc_names))
OTHER_LR_mean=np.zeros(len(unassoc_names))
OTHER_LR_std=np.zeros(len(unassoc_names))
#########################





########################################
for i in range(len(unassoc_names)):
    AGN_BDT=[]
    PSR_BDT=[]
    OTHER_BDT=[]
    AGN_NN=[]
    PSR_NN=[]
    OTHER_NN=[]
    AGN_RF=[]
    PSR_RF=[]
    OTHER_RF=[]
    AGN_LR=[]
    PSR_LR=[]
    OTHER_LR=[]
    for j in range(times):
        prob=probs_un_bdt[j]
        prob2=prob[i]
        AGN_BDT.append(prob2[0])
        PSR_BDT.append(prob2[1])
        #OTHER_BDT.append(prob2[1])
        prob=probs_un_nn[j]
        prob2=prob[i]
        AGN_NN.append(prob2[0])
        PSR_NN.append(prob2[1])
        #OTHER_NN.append(prob2[1])
        prob=probs_un_rf[j]
        prob2=prob[i]
        AGN_RF.append(prob2[0])
        PSR_RF.append(prob2[1])
        #OTHER_RF.append(prob2[1])
        prob=probs_un_lr[j]
        prob2=prob[i]
        AGN_LR.append(prob2[0])
        PSR_LR.append(prob2[1])
        #OTHER_LR.append(prob2[1])





    AGN_BDT_mean[i]=np.mean(AGN_BDT)
    AGN_BDT_std[i]=np.std(AGN_BDT)
    PSR_BDT_mean[i]=np.mean(PSR_BDT)
    PSR_BDT_std[i]=np.std(PSR_BDT)
    #OTHER_BDT_mean[i]=np.mean(OTHER_BDT)
    #OTHER_BDT_std[i]=np.std(OTHER_BDT)
    AGN_NN_mean[i]=np.mean(AGN_NN)
    AGN_NN_std[i]=np.std(AGN_NN)
    PSR_NN_mean[i]=np.mean(PSR_NN)
    PSR_NN_std[i]=np.std(PSR_NN)
    #OTHER_NN_mean[i]=np.mean(OTHER_NN)
    #OTHER_NN_std[i]=np.std(OTHER_NN)
    AGN_RF_mean[i]=np.mean(AGN_RF)
    AGN_RF_std[i]=np.std(AGN_RF)
    PSR_RF_mean[i]=np.mean(PSR_RF)
    PSR_RF_std[i]=np.std(PSR_RF)
    #OTHER_RF_mean[i]=np.mean(OTHER_RF)
    #OTHER_RF_std[i]=np.std(OTHER_RF)
    AGN_LR_mean[i]=np.mean(AGN_LR)
    AGN_LR_std[i]=np.std(AGN_LR)
    PSR_LR_mean[i]=np.mean(PSR_LR)
    PSR_LR_std[i]=np.std(PSR_LR)
    #OTHER_LR_mean[i]=np.mean(OTHER_LR)
    #OTHER_LR_std[i]=np.std(OTHER_LR)

    



#result2=np.column_stack((unassoc_names,AGN_BDT_mean,AGN_BDT_std,PSR_BDT_mean,PSR_BDT_std,OTHER_BDT_mean,OTHER_BDT_std,AGN_NN_mean,AGN_NN_std,PSR_NN_mean,PSR_NN_std,OTHER_NN_mean,OTHER_NN_std,AGN_RF_mean,AGN_RF_std,PSR_RF_mean,PSR_RF_std,OTHER_RF_mean,OTHER_RF_std,AGN_LR_mean,AGN_LR_std,PSR_LR_mean,PSR_LR_std,OTHER_LR_mean,OTHER_LR_std))
result2=np.column_stack((unassoc_names,AGN_BDT_mean,AGN_BDT_std,PSR_BDT_mean,PSR_BDT_std,AGN_NN_mean,AGN_NN_std,PSR_NN_mean,PSR_NN_std,AGN_RF_mean,AGN_RF_std,PSR_RF_mean,PSR_RF_std,AGN_LR_mean,AGN_LR_std,PSR_LR_mean,PSR_LR_std))
pro2=["Source_Name","AGN_BDT","AGN_BDT_STD","PSR_BDT","PSR_BDT_STD","AGN_NN","AGN_NN_STD","PSR_NN","PSR_NN_STD","AGN_RF","AGN_RF_STD","PSR_RF","PSR_RF_STD","AGN_LR","AGN_LR_STD","PSR_LR","PSR_LR_STD"]

#pro2=["Source_Name","AGN_BDT","AGN_BDT_STD","PSR_BDT","PSR_BDT_STD","OTHER_BDT","OTHER_BDT_STD","AGN_NN","AGN_NN_STD","PSR_NN","PSR_NN_STD","OTHER_NN","OTHER_NN_STD","AGN_RF","AGN_RF_STD","PSR_RF","PSR_RF_STD","OTHER_RF","OTHER_RF_STD","AGN_LR","AGN_LR_STD","PSR_LR","PSR_LR_STD","OTHER_LR","OTHER_LR_STD"]
result=np.vstack((pro2,result2))
result=pandas.DataFrame(result)

result.to_csv(path_or_buf="./catas/try_3fgl_missingv_O.csv",index=False)
