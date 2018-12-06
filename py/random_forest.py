import numpy as np
from random import shuffle, sample
from astropy.io import fits
from numpy import vstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

seed=5
np.random.seed(seed)
file_1=fits.open('input3')
file_2=fits.open('OUTPUT')
file_3=fits.open('input_glat')



input_1  = np.empty((1905,5))
input_1=vstack(file_1[1].data)
#print(input_1.shape)
input_2=np.empty((1905,6))
input_2=vstack(file_3[1].data)

output_1 = np.empty((1905,1))
output_1=vstack(file_2[1].data)

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
num=15
input_1=np.append(input_1,output_1,axis=1)    #append labels with inputs
input_2=np.append(input_2,output_1,axis=1)

np.random.shuffle(input_1)                    #shuffle the entire array
np.random.shuffle(input_2)


    
    
print(num)
#divide into training and testing:
train1=input_1[:1520,0:5]                    
train_truth1=input_1[:1520,5:]
val_inp1=input_1[1520:,:5]
val_out1=input_1[1520:,5:]
val_out1=np.ravel(val_out1)                     #ravel is used since flattened label array required
train_truth1=np.ravel(train_truth1)

train2=input_2[:1520,0:6]                    
train_truth2=input_2[:1520,6:]
val_inp2=input_2[1520:,:6]
val_out2=input_2[1520:,6:]
val_out2=np.ravel(val_out2)                     #ravel is used since flattened label array required
train_truth2=np.ravel(train_truth2)

#print(train_truth.shape)
#for i in range(1400):
#    if train_truth[i]=="AGN":
#        output_1[i,:]=1
#    else:
#        output_1[i,:]=-1
#        num+=1
#print(num)
#test_inp=input_1[1300:1600,:5]
#test_out=input_1[1300:1600,5:]
i=10
j=2
numi=[]
numj=[]
oobscore=[]
valscore=[]
oobscore2=[]
valscore2=[]
while i < 300:
    while j < 20:
        clf = RandomForestClassifier(n_estimators=i,max_depth=j,oob_score=True,random_state=0,class_weight='balanced')
        clf.fit(train1,train_truth1)
        numi.append(i)
        numj.append(j)
        oobscore.append(clf.oob_score_)
        valscore.append(clf.score(val_inp1,val_out1))
        clf.fit(train2,train_truth2)
        oobscore2.append(clf.oob_score_)
        valscore2.append(clf.score(val_inp2,val_out2))
        print(i)
        print(j)
        j=j+3
    i=i+40
    j=2

fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(numi, numj, oobscore, 'o')
ax.set_xlabel('trees')
ax.set_ylabel('max depth')
ax.set_zlabel('oob score')
ax.set_title('for without glat')
plt.show()

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
clf = RandomForestClassifier(n_estimators=50,max_depth=num,oob_score=True,random_state=0,class_weight='balanced') #n_estimators=no. of trees
clf1 = ExtraTreesClassifier(n_estimators=50,oob_score=True,bootstrap=True,random_state=0)
clf2 = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,max_depth=num, random_state=0).fit(train1, train_truth1)
clf3 = AdaBoostClassifier(n_estimators=50,random_state=0)
clf4 = LogisticRegression(random_state=0, solver='saga').fit(train1, train_truth1)
eclf = VotingClassifier(estimators=[('rf', clf), ('et', clf1), ('gb', clf2),('Ab',clf3),('LR',clf4)], voting='hard')    #used for the ensemble classification

clf.fit(train1,train_truth1)                                      #fit model
clf1.fit(train1,train_truth1)
clf3.fit(train1,train_truth1)
eclf.fit(train1,train_truth1)

#score1.append(clf.oob_score_)
#score1.append(clf1.oob_score_)
#score1.append(clf2.oob_score_)
#score1.append(clf3.oob_score_)
#score1.append(eclf.oob_score_)
score1.append(clf.score(val_inp1,val_out1))
score1.append(clf1.score(val_inp1,val_out1))
score1.append(clf2.score(val_inp1,val_out1))
score1.append(clf3.score(val_inp1,val_out1))
score1.append(eclf.score(val_inp1,val_out1))

clf2 = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,max_depth=num, random_state=0).fit(train2, train_truth2)
clf.fit(train2,train_truth2)                                      #fit model
clf1.fit(train2,train_truth2)
clf3.fit(train2,train_truth2)
clf4 = LogisticRegression(random_state=0, solver='saga').fit(train2, train_truth2)
eclf.fit(train2,train_truth2)

#score2.append(clf.oob_score_)
#score2.append(clf1.oob_score_)
#score2.append(clf2.oob_score_)
#score2.append(clf3.oob_score_)
#score2.append(eclf.oob_score_)




score2.append(clf.score(val_inp2,val_out2))
score2.append(clf1.score(val_inp2,val_out2))
score2.append(clf2.score(val_inp2,val_out2))
score2.append(clf3.score(val_inp2,val_out2))
score2.append(eclf.score(val_inp2,val_out2))

labels = ['rf','ET','GB','ABC','Vot']

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
