
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

pyplot.rcParams['xtick.labelsize'] = 18
pyplot.rcParams['axes.labelsize'] = 18
pyplot.rcParams['axes.titlesize'] = 28
pyplot.rcParams['font.size'] = 21
pyplot.rcParams['ytick.labelsize'] = 18
se=0
valscore3=np.zeros(10)
valscore4=np.zeros(10)
valscore12=np.zeros(10)
valscore22=np.zeros(10)
feat=np.zeros(10)
feat2=np.zeros(10)
while se<100:
    
    np.random.seed(se)
    #dataframe = pandas.read_csv("4fgl_assoc_3.csv", header=None)
    dataframe = pandas.read_csv("./files/3fgl_associated_AGNandPSR.csv", header=None)
    dataset1 = dataframe.values 
    np.random.shuffle(dataset1[1:])
#labels2=dataset[:1,:5]
#print(dataset)
# split into input (X) and output (Y) variables
#X = dataset[1:1933,0:5].astype(float)
    X = dataset1[1:,0:10].astype(float)
    print(X)
    #print(dataset1[2,:])
#Y = dataset[1:1933,5]
    Y = dataset1[1:,10]
    print(se)
    
#dataset=dataset[:,:5]
    num=[]
    num2=[]
    resul=[]

    weight1=800/166
    weight2=800/1739
    class_weights= {'PSR': weight1, 'AGN': weight2}
# encode class values as integers

    encoder = preprocessing.LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)
    #print(Y)
    '''
    dataframe = pandas.read_csv("3fgl_unass_withclasses_nn_allfeat.csv", header=None)
    dataset = dataframe.values
    X2 = dataset[1:,0:10].astype(float)
    #   print(len(X2))
    Y2 = dataset[1:,10]
    encoder = preprocessing.LabelEncoder()
    encoder.fit(Y2)
    Y2 = encoder.transform(Y2)
#num=0
#for i in range(1905):
#    if output_1[i,:]=="AGN":
#        output_1[i,:]=1
#    else:
#        output_1[i,:]=-1
#        num+=1
#print(num)
    score1=[]
    score2=[]
    num=11
    '''
#Y = dataset[1:1933,5]



    
    
#divide into training and testing:
    #train1=X[0:1500]                    
    #train_truth1=Y[0:1500]
    #val_inp1=X[1500:]
    #val_out1=Y[1500:]
    train1,val_inp1, train_truth1,  val_out1 = train_test_split(X, Y, test_size=.3, random_state=se)       #Split into training and validation

    val_out1=np.ravel(val_out1)                     #ravel is used since flattened label array required
    train_truth1=np.ravel(train_truth1)



    i=1
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
    valscore2=valscore3
    valscore6=valscore4
    valscore11=valscore12
    valscore21=valscore22
    while i < 21:
        #clf = RandomForestClassifier(n_estimators=20,max_depth=i,oob_score=True,random_state=0,class_weight="balanced")
        #clf.fit(train1,train_truth1)
        clf = MLPClassifier(max_iter=300,hidden_layer_sizes=(10,i,), activation='tanh', solver='lbfgs').fit(train1,train_truth1)






        #clf=GradientBoostingClassifier(n_estimators=20, learning_rate=0.3,max_depth=i, random_state=0).fit(train1, train_truth1)
        #clf= LogisticRegression(max_iter=i, C=1, solver='lbfgs').fit(train1, train_truth1)
        numi.append(i)
        
        scor=clf.score(val_inp1,val_out1)
        valscore.append(scor*100)
        
        #feat3.append(clf.feature_importances_)
        #print(scor)
        #print(i)
        clf2 = MLPClassifier(max_iter=300,hidden_layer_sizes=(10,i,), activation='relu', solver='lbfgs').fit(train1,train_truth1)
        #clf2=GradientBoostingClassifier(n_estimators=50, learning_rate=0.3,max_depth=i, random_state=0).fit(train1, train_truth1)
        #clf2 = RandomForestClassifier(n_estimators=50,max_depth=i,oob_score=True,random_state=0,class_weight="balanced")
        #clf2.fit(train1,train_truth1)

        #clf2= LogisticRegression(max_iter=i, C=1, solver='liblinear').fit(train1, train_truth1)
        #clf3=GradientBoostingClassifier(n_estimators=100, learning_rate=0.3,max_depth=i, random_state=0).fit(train1, train_truth1)
        clf3 = MLPClassifier(max_iter=300,hidden_layer_sizes=(10,i,), activation='tanh', solver='adam').fit(train1,train_truth1)
        score2=clf2.score(val_inp1,val_out1)
        #clf3 = RandomForestClassifier(n_estimators=100,max_depth=i,oob_score=True,random_state=0,class_weight="balanced")
        #clf3.fit(train1,train_truth1)

        valscore5.append(score2*100)
        #clf3= LogisticRegression(max_iter=i, C=1,solver='sag').fit(train1, train_truth1)
        score3=clf3.score(val_inp1,val_out1)
        valscore10.append(score3*100)
        clf4 = MLPClassifier(max_iter=300,hidden_layer_sizes=(10,i,), activation='relu', solver='adam').fit(train1,train_truth1)
        #clf4=GradientBoostingClassifier(n_estimators=500, learning_rate=0.3,max_depth=i, random_state=0).fit(train1, train_truth1)
        #clf4 = RandomForestClassifier(n_estimators=200,max_depth=i,oob_score=True,random_state=0, class_weight="balanced")
        #clf4.fit(train1,train_truth1)
        #clf4= LogisticRegression(max_iter=i, C=1,solver='saga').fit(train1, train_truth1)

        score4=clf4.score(val_inp1,val_out1)
        valscore20.append(score4*100)
        
        

        
            
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
#print(feat)
valscore3=valscore3/100

valscore4=valscore4/100
valscore12=valscore12/100
valscore22=valscore22/100
print(valscore3)
print(valscore4)
print(valscore12)
print(valscore22)

fig,ax=plt.subplots()
#print(valscore3)
#ax = fig.add_subplot(111, projection='3d')
plt.plot(numi, valscore3, 'g--',marker='o')
plt.plot(numi, valscore4, 'b--',marker='o')
plt.plot(numi, valscore12, 'r--',marker='o')
plt.plot(numi, valscore22, 'm--',marker='x')

#ax.set_xlabel('Regularization Parameter',fontsize='xx-large')
ax.set_xlabel('Number of Neurons in second hidden layer',fontsize='xx-large')

ax.set_ylabel('Testing Score',fontsize='xx-large')
plt.yticks(fontsize='large')
#plt.yticks(np.arange(92,99,step=1))
#plt.xticks(fontsize='large')
#ax.set_zlabel('Validation score')
#plt.legend(["20 Trees","50 Trees","100 Trees","200 Trees"])
#plt.legend(["Tol= 0.001","Tol = 1","Tol = 10"])
plt.legend(["LBFGS Tanh","LBFGS Relu","ADAM Tanh","ADAM Relu"])

#ax.set_title('Logistic Regression (LBFGS,300): Accuracy vs. Regularization',fontsize='xx-large')
ax.set_title('Neural Networks: 2 Hidden Layers',fontsize='xx-large')

plt.show()

result=numi
result=np.vstack((result,valscore3))
result=np.vstack((result,valscore4))
result=np.vstack((result,valscore12))
result=np.vstack((result,valscore22))
print(result)
result=pandas.DataFrame(result)
result.to_csv(path_or_buf="./files/result_3fglassoc_nn_2layers.csv",index=False)

'''
    i=0
    feat=feat2
    valscore2=valscore3
    clfNN=MLPClassifier(max_iter=50,hidden_layer_sizes=(20,)).fit(train1,train_truth1)
    clf = RandomForestClassifier(n_estimators=50,max_depth=12,oob_score=True,random_state=0,class_weight="balanced")
    clf2 = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,max_depth=15, random_state=0).fit(train1, train_truth1)
    clf2.fit(train1,train_truth1)
    clf.fit(train1,train_truth1)
    clf4 = LogisticRegression(random_state=0,class_weight='balanced', solver='liblinear', max_iter=300, penalty='l2').fit(train1, train_truth1)
    valscore=clf2.score(val_inp1,val_out1)
    valscore3=(valscore+valscore2)/2
    feat1=clf4.coef_[0]
    print(feat1)
    for i in range(10):
        feat2[i]=(feat[i]+feat1[i])/2
        i=i+1
    se=se+1
    print(feat2)

'''














    
'''
    clf = RandomForestClassifier(n_estimators=50,max_depth=num,oob_score=True,random_state=0,class_weight="balanced") #n_estimators=no. of trees
    clf1 = ExtraTreesClassifier(n_estimators=50,oob_score=True,bootstrap=True,random_state=0)
    clf2 = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,max_depth=num, random_state=0).fit(train1, train_truth1)
    clf3 = AdaBoostClassifier(n_estimators=50,random_state=0)
    clf4 = LogisticRegression(random_state=0,class_weight='balanced', solver='liblinear').fit(train1, train_truth1)
    eclf = VotingClassifier(estimators=[('rf', clf), ('et', clf1), ('gb', clf2),('Ab',clf3),('LR',clf4)], voting='hard')
    clf.fit(train1,train_truth1)                                      #fit model
    clf1.fit(train1,train_truth1)
    clf3.fit(train1,train_truth1)
    eclf.fit(train1,train_truth1)
#print(clf.score(train1[1500:,:],train_truth1[1500:]))
    
    print(clf.score(val_inp1,val_out1))
    print(clf1.score(val_inp1,val_out1))
    print(clf2.score(val_inp1,val_out1))
    print(clf3.score(val_inp1,val_out1))
    print(clf4.score(val_inp1,val_out1))
''' 



'''
d=clf.predict(val_inp1)
l=0
print(len(d))
for i in range(len(d)):
    if d[i]==1:
        l=l+1
print(l)
'''
'''
    i=50
    j=1

    numi=[]
    numj=[]
    oobscore=[]
    valscore=[]
    oobscore2=[]
    valscore2=valscore3
    while i < 100:
        while j < 122:
        #clf=GradientBoostingClassifier(n_estimators=i, learning_rate=0.1,max_depth=j, random_state=0).fit(train1, train_truth1)
            #clf = RandomForestClassifier(n_estimators=i,max_depth=j,oob_score=True,random_state=0,class_weight='balanced')
            clf=MLPClassifier(max_iter=j,hidden_layer_sizes=(5,))
            clf.fit(train1,train_truth1)
            numi.append(i)
            numj.append(j)
        #oobscore.append(clf.oob_score_)
            valscore.append(clf.score(val_inp1,val_out1)*100)
                
        #print(i)
        #print(j)
            j=j+3
        i=i+100
        j=1
        print(i)
    #k=0
    #for k in range(len(valscore)):
        
        valscore=(valscore2+valscore)/2
     #   k=k+1
'''
'''
fig,ax=plt.subplots()
#ax = fig.add_subplot(111, projection='3d')
plt.plot(numj, valscore3, 'r--',marker='o')
ax.set_xlabel('Number of Epochs',fontsize='xx-large')
ax.set_ylabel('Validation Score',fontsize='xx-large')
plt.yticks()
#plt.yticks(np.arange(92,99,step=1))
plt.xticks(np.arange(0,121,step=10),fontsize='xx-large')
#ax.set_zlabel('Validation score')
ax.set_title('Testing Score for Neural Networks with 1 Hidden Layer (5 neurons)',fontsize='xx-large')
plt.show()
'''
'''
fig2=plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
plt.plot(numi, numj, valscore, 'o')
ax2.set_xlabel('trees')
ax2.set_ylabel('max depth')
ax2.set_zlabel('val score')
ax2.set_title('for without glat')
plt.show()

fig3=plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
plt.plot(numi, numj, oobscore2, 'o')
ax3.set_xlabel('trees')
ax3.set_ylabel('max depth')
ax3.set_zlabel('oob score')
ax3.set_title('for with glat')
plt.show()

fig4=plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
plt.plot(numi, numj, valscore2, 'o')
ax4.set_xlabel('trees')
ax4.set_ylabel('max depth')
ax4.set_zlabel('val score')
ax4.set_title('for with glat')
plt.show()
'''
'''
clf = RandomForestClassifier(n_estimators=50,max_depth=num,oob_score=True,random_state=0,class_weight="balanced") #n_estimators=no. of trees
clf1 = ExtraTreesClassifier(n_estimators=50,oob_score=True,bootstrap=True,random_state=0)
clf2 = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,max_depth=num, random_state=0).fit(train1, train_truth1)
clf3 = AdaBoostClassifier(n_estimators=50,random_state=0)
clf4 = LogisticRegression(random_state=0,class_weight='balanced', solver='liblinear').fit(train1, train_truth1)
eclf = VotingClassifier(estimators=[('rf', clf), ('et', clf1), ('gb', clf2),('Ab',clf3),('LR',clf4)], voting='hard')    #used for the ensemble classification

clf.fit(train1,train_truth1)                                      #fit model
clf1.fit(train1,train_truth1)
clf3.fit(train1,train_truth1)
eclf.fit(train1,train_truth1)
#print(clf.score(train1[1500:,:],train_truth1[1500:]))
print(dataset1[:1])
print(clf.score(val_inp1,val_out1))
print(clf1.score(val_inp1,val_out1))
print(clf2.score(val_inp1,val_out1))
print(clf3.score(val_inp1,val_out1))
print(clf4.score(val_inp1,val_out1))
#Predictions:
'''
#pro=clf.predict(val_inp1)
#print(pro)
#count=0
'''
for i in range(286):
    if pro[i]=="PSR": count=count+1
print(count)
'''

'''
print(clf.feature_importances_)
print(clf1.feature_importances_)
print(clf2.feature_importances_)
print(clf3.feature_importances_)
print(clf4.feature_importances_)
'''

#pro=pandas.DataFrame(pro)
#pro.to_csv(path_or_buf="RF_predict_newind_1_286sources_3feat_unrandom.csv",index=False)



#classifier comparison:
'''
linearly_separable = (X, Y)
datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]
print(datasets)
#linearly_separable = (X, Y)
#datasets = [make_moons(noise=0.3, random_state=0),
#            make_circles(noise=0.2, factor=0.5, random_state=1),
#            linearly_separable
#            ]

print(len(datasets))
names = [ "Linear SVM", "RBF SVM","Logistic regression","Decision Tree", "Random Forest", "Neural Net", "AdaBoost","Gradient Boost"]
classifiers = [
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    LogisticRegression(random_state=0,class_weight='balanced', solver='liblinear'),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(n_estimators=50,max_depth=num,oob_score=True,random_state=0,class_weight="balanced"),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(n_estimators=50,random_state=0),
    GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,max_depth=num, random_state=0)]
h=.02
figure = plt.figure(figsize=(27, 9))
i = 1
for ds_cnt, ds in enumerate(datasets):
#train_truth1 = StandardScaler().fit_transform(train_truth1)
    X, y = ds
    X_train = train1
    y_train=train_truth1
    X_test=val_inp1
    y_test=val_out1
    #y_train, y_test = train_test_split(train1, train_truth1, test_size=.3, random_state=42)

    x_min, x_max = train1[:, 0].min() - .5, train1[:, 0].max() + .5
    y_min, y_max = train1[:, 1].min() - .5, train1[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets),len(classifiers) + 1,i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets),len(classifiers)+1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(name)
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max()- .3, yy.max() - .3, ('%.2f' % score).lstrip('0'),size=10, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()
#score1.append(clf.oob_score_)
#score1.append(clf1.oob_score_)
#score1.append(clf2.oob_score_)
#score1.append(clf3.oob_score_)
#score1.append(eclf.oob_score_)
'''
'''

print(clf4.score(val_inp1,val_out1))
score1.append(clf.score(val_inp1,val_out1))
score1.append(clf1.score(val_inp1,val_out1))
score1.append(clf2.score(val_inp1,val_out1))
score1.append(clf3.score(val_inp1,val_out1))
score1.append(clf4.score(val_inp1,val_out1))
score1.append(eclf.score(val_inp1,val_out1))
print(clf1.feature_importances_)
print(score1)
'''


'''
dataset=pandas.DataFrame(X)
plt.rc('xtick', labelsize=7) 
plt.rc('ytick', labelsize=10)
sai=dataset.corr(method='pearson')

print(sai)

axe=sns.heatmap(sai,annot=True,xticklabels=False, yticklabels=labels2)

plt.show()
'''




'''
clf2 = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,max_depth=num, random_state=0).fit(train2, train_truth2)
clf.fit(train2,train_truth2)                                      #fit model
clf1.fit(train2,train_truth2)
clf3.fit(train2,train_truth2)
clf4 = LogisticRegression(random_state=0, solver='liblinear',class_weight='balanced').fit(train2, train_truth2)
eclf.fit(train2,train_truth2)
'''
#score2.append(clf.oob_score_)
#score2.append(clf1.oob_score_)
#score2.append(clf2.oob_score_)
#score2.append(clf3.oob_score_)
#score2.append(eclf.oob_score_)



'''
score2.append(clf.score(val_inp2,val_out2))
score2.append(clf1.score(val_inp2,val_out2))
score2.append(clf2.score(val_inp2,val_out2))
score2.append(clf3.score(val_inp2,val_out2))
score1.append(clf4.score(val_inp2,val_out2))
score2.append(eclf.score(val_inp2,val_out2))
'''

'''
labels = ['rf','ET','GB','ABC','LR','Vot']

fig,ax=plt.subplots()
plt.subplots_adjust(bottom = 0.1)
plt.scatter(score1, score2, marker='o', cmap=plt.get_cmap('Spectral'))

for label, x, y in zip(labels, score1, score2):
    plt.annotate(label,xy=(x, y), xytext=(-10, 10),textcoords='offset points', ha='right', va='bottom',bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5),arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

#plt.plot(score1,score2,'o', markersize=10)
ax.set_xlabel('validation score without glat')
ax.set_ylabel('validation score with glat')
ax.set_title('score vs. score with weights and 1 shuffle for glat')
plt.show()
'''
'''
use the below part if you want to use an ensemble of the above classifiers
eclf.fit(train,train_truth)
for clf, label in zip([clf, clf1, clf2,clf3, eclf], ['Rf', 'Et', 'GB','Adab', 'Ensemble']):
    scores = cross_val_score(clf, val_inp, val_out, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

print("oob scores(Rf,ET):")
print(clf.oob_score_)
print(clf1.oob_score_)

#print(clf.classes_)
k=clf.feature_importances_
print("feature importance (flux,unc,signif_curve,spectral_index,var):")
print(k)
print(clf1.feature_importances_)
print(clf2.feature_importances_)
print(clf3.feature_importances_)
print("score for validation(Rf,ET,GB,AdaB,total):")
print(clf.score(val_inp,val_out))
print(clf1.score(val_inp,val_out))
print(clf2.score(val_inp,val_out))
print(clf3.score(val_inp,val_out))
#print(eclf.score(val_inp,val_out))
'''

'''
for i in range(len(train_truth)):
    if train_truth[i]=="AGN":
        train_truth[i]=1
    else:
        train_truth[i]=0

for i in range(len(val_out)):
    if val_out[i]=="AGN":
        val_out[i]=1
    else:
        val_out[i]=0
clf4 = LogisticRegression(random_state=0, solver='saga').fit(train, train_truth)
print(clf4.score(val_inp,val_out))


'''
