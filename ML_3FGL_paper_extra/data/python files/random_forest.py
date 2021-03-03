  
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


pyplot.rcParams['xtick.labelsize'] = 18
pyplot.rcParams['axes.labelsize'] = 18
pyplot.rcParams['axes.titlesize'] = 28
pyplot.rcParams['font.size'] = 21
pyplot.rcParams['ytick.labelsize'] = 18
se=0
size=10
valscore3=np.zeros(size)
valscore4=np.zeros(size)
valscore12=np.zeros(size)
valscore22=np.zeros(size)
feat=0
feat2=np.zeros(10)
lenth=17
while se<10:
    
    np.random.seed(se)
    dataframe = pandas.read_csv("./files/4fgldr2_all_newfeats.csv", header=None)
    #dataframe = pandas.read_csv("./files/3fgl_all_newfeats.csv", header=None)
    dataset1 = dataframe.values[1:]
    k=dataset1[1:,1].astype(float)
    dataset1[1:,1]=np.cos(k)
    np.random.shuffle(dataset1[:])
    #######################################################################################
    X=[dataset1[i,1:(lenth)].astype(float) for i in range(len(dataset1)) if dataset1[i,lenth]!='UNAS']# or dataset1[i,lenth]=='PSR']# or dataset1[i,lenth]=='OTHER']
    Y =[dataset1[i,(lenth+1)] for i in range(len(dataset1)) if dataset1[i,lenth]!='UNAS']# or dataset1[i,lenth]=='PSR']#or dataset1[i,lenth]=='OTHER']
    #print((Y))
    #kutta
    X = StandardScaler(with_mean=False,with_std=False).fit_transform(X)
    encoder = preprocessing.LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)
    #print(Y)
    #kutta
    ############################################################################################
    
    #divide into training and testing:
    train1,val_inp1, train_truth1,  val_out1 = train_test_split(X, Y, test_size=.3, random_state=se)       #Split into training and validation
    val_out1=np.ravel(val_out1)                     #ravel is used since flattened label array required
    train_truth1=np.ravel(train_truth1)

    #sampling_strategy = 'not majority'
    #oversample = RandomOverSampler(sampling_strategy=weight)
    #X_over, y_over = oversample.fit_resample(train1, train_truth1)
    #ros = RandomOverSampler(sampling_strategy=sampling_strategy)
    #X_over, y_over = ros.fit_resample(train1, train_truth1)
    X_over, y_over=train1,train_truth1

    ###################################################################################################
    i=2
    j=0
    feat3=[]
    feat2=feat
    numi=[]
    numj=[]
    oobscore=[]
    valscore=[]
    valscore5=[]
    valscore10=[]
    valscore20=[]
    oobscore2=[]
    neur=16
    valscore2=valscore3
    valscore6=valscore4
    valscore11=valscore12
    valscore21=valscore22
    while i < 21:
        #clf = RandomForestClassifier(n_estimators=20,max_depth=i,oob_score=True)
        #clf.fit(train1,train_truth1)
        clf = MLPClassifier(max_iter=500,hidden_layer_sizes=(neur,i,), activation='tanh', solver='adam').fit(X_over, y_over)
        #clf= GradientBoostingClassifier(n_estimators=5, learning_rate=0.3,max_depth=i).fit(X_over, y_over)
        #clf= LogisticRegression(max_iter=i, C=1, solver='lbfgs').fit(X_over, y_over)
        numi.append(i)
        scor=clf.score(val_inp1,val_out1)

        #feat=feat+clf.feature_importances_
        #print(feat/(se+1))
        valscore.append(scor*100)
        
        #feat3.append(clf.feature_importances_)
        #print(i)
        clf2 = MLPClassifier(max_iter=1000,hidden_layer_sizes=(neur,i,), activation='tanh', solver='adam').fit(X_over, y_over)
        #clf2=GradientBoostingClassifier(n_estimators=600, learning_rate=0.3,max_depth=i).fit(X_over, y_over)
        #clf2 = RandomForestClassifier(n_estimators=50,max_depth=i+4,oob_score=True)
        #clf2.fit(train1,train_truth1)

        #clf2= LogisticRegression(max_iter=i, C=1, solver='liblinear').fit(X_over, y_over)
        #clf3=GradientBoostingClassifier(n_estimators=100, learning_rate=0.3,max_depth=i).fit(X_over, y_over)
        clf3 = MLPClassifier(max_iter=1500,hidden_layer_sizes=(neur,i,), activation='tanh', solver='adam').fit(X_over, y_over)
        score2=clf2.score(val_inp1,val_out1)
        #clf3 = RandomForestClassifier(n_estimators=100,max_depth=i,oob_score=True)
        #clf3.fit(train1,train_truth1)

        valscore5.append(score2*100)
        #clf3= LogisticRegression(max_iter=i, C=1,solver='sag').fit(X_over, y_over)
        score3=clf3.score(val_inp1,val_out1)
        valscore10.append(score3*100)
        #print(score3)
        clf4 = MLPClassifier(max_iter=2000,hidden_layer_sizes=(neur,i,), activation='tanh', solver='adam').fit(X_over, y_over)
        #clf4=GradientBoostingClassifier(n_estimators=500, learning_rate=0.3,max_depth=i).fit(X_over, y_over)
        #clf4 = RandomForestClassifier(n_estimators=200,max_depth=i,oob_score=True)
        #clf4.fit(train1,train_truth1)
        #clf4= LogisticRegression(max_iter=i, C=1,solver='saga').fit(X_over, y_over) 

        score4=clf4.score(val_inp1,val_out1)
        valscore20.append(score4*100)
        
        #print('scores for {} iterations: lbfgst {},lbfgsr {},adamt {},adamr {}'.format(i,scor*100,score2*100,score3*100,score4*100))


        
            
        i=i+2
        
        
    #k=0
    #for k in range(len(valscore)):
    #lent=len(valscore)
    #feat=(feat3+feat2)
    valscore3=valscore2+valscore
    valscore4=valscore5+valscore6
    valscore12=valscore11+valscore10
    valscore22=valscore21+valscore20
    se=se+1
    print(se)
#si=clf.coefs_
#feat=feat/100
#print("feat(BDT)",dataframe.values[0,1:11],valscore4/100)
#print("feat(RF)",dataframe.values[0,1:11],valscore12/100)

valscore3=valscore3/10
valscore4=valscore4/10
valscore12=valscore12/10
valscore22=valscore22/10
#print(valscore3)
#print(valscore4)
#print(valscore12)
#print(valscore22)

fig,ax=plt.subplots()
#print(valscore3)
#ax = fig.add_subplot(111, projection='3d')
plt.plot(numi, valscore3, 'g--',marker='o')
plt.plot(numi, valscore4, 'b--',marker='o')
plt.plot(numi, valscore12, 'r--',marker='o')
plt.plot(numi, valscore22, 'm--',marker='x')

#ax.set_xlabel('Regularization Parameter',fontsize='xx-large')
#ax.set_xlabel('Maximum Depth',fontsize='xx-large')
#ax.set_xlabel('Epochs',fontsize='xx-large')
ax.set_xlabel('Number of Neurons in 2nd Layer',fontsize='xx-large')

ax.set_ylabel('Testing Score',fontsize='xx-large')
plt.yticks(fontsize='large')
#plt.yticks(np.arange(92,99,step=1))
#plt.xticks(fontsize='large')
#ax.set_zlabel('Validation score')
#plt.legend(["20 Trees","50 Trees","100 Trees","200 Trees"])
#plt.legend(["Tol= 0.001","Tol = 1","Tol = 10"])
plt.legend(["500 Epochs","1000 Epochs","1500 Epochs","2000 Epochs"],title=("Number of Neurons: {}".format(neur)))
#plt.legend(["LBFGS","Liblinear","SAG","SAGA"])

#ax.set_title('Logistic Regression (LBFGS,300): Accuracy vs. Regularization',fontsize='xx-large')
#ax.set_title('Random Forests',fontsize='xx-large')
ax.set_title('Neural Network',fontsize='xx-large')

plt.show()

result=numi
result=np.vstack((result,valscore3))
result=np.vstack((result,valscore4))
result=np.vstack((result,valscore12))
result=np.vstack((result,valscore22))
print(result)
result=pandas.DataFrame(result)
#result.to_csv(path_or_buf="./files/result_3fglassocnewfeat_nn_epochs_multi_cosglon_oversampled.csv",index=False)



















