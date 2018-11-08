import numpy as np
import matplotlib.pyplot as plt
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

file_1=fits.open('input3')
file_2=fits.open('OUTPUT')

input_1  = np.empty((1905,5))
input_1=vstack(file_1[1].data)
#print(input_1.shape)

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
input_1=np.append(input_1,output_1,axis=1)    #append labels with inputs
np.random.shuffle(input_1)                    #shuffle the entire array


#divide into training and testing:
train=input_1[:1400,0:5]                    
train_truth=input_1[:1400,5:]
val_inp=input_1[1400:,:5]
val_out=input_1[1400:,5:]
val_out=np.ravel(val_out)                     #ravel is used since flattened label array required
train_truth=np.ravel(train_truth)


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

clf = RandomForestClassifier(n_estimators=10000,oob_score=True,random_state=0) #n_estimators=no. of trees
clf1 = ExtraTreesClassifier(n_estimators=10000,oob_score=True,bootstrap=True,random_state=0)
clf2 = GradientBoostingClassifier(n_estimators=10000, learning_rate=1.0,max_depth=1, random_state=0).fit(train, train_truth)
clf3 = AdaBoostClassifier(n_estimators=1000,random_state=0)
#eclf = VotingClassifier(estimators=[('rf', clf), ('et', clf1), ('gb', clf2),('Ab',clf3)], voting='hard')    #used for the ensemble classification

clf.fit(train,train_truth)                                      #fit model
clf1.fit(train,train_truth)
clf3.fit(train,train_truth)


'''    use the below part if you want to use an ensemble of the above classifiers
eclf.fit(train,train_truth)
for clf, label in zip([clf, clf1, clf2,clf3, eclf], ['Rf', 'Et', 'GB','Adab', 'Ensemble']):
    scores = cross_val_score(clf, val_inp, val_out, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
'''
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
