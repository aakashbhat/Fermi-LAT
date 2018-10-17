import numpy as np
import matplotlib.pyplot as plt
from random import shuffle, sample
from astropy.io import fits
from numpy import vstack
from sklearn.ensemble import RandomForestClassifier

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

clf = RandomForestClassifier(n_estimators=10000,oob_score=True) #n_estimators=no. of trees
clf.fit(train,train_truth)                                      #fit model
print("oob score:")
print(clf.oob_score_)
#print(clf.classes_)
k=clf.feature_importances_
print("feature importance (flux,unc,signif_curve,spectral_index,var):")
print(k)
print("score for validation:")
print(clf.score(val_inp,val_out))
