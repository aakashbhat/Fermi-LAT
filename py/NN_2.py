import numpy
import pandas
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

seed = 5
numpy.random.seed(seed)

# load dataset
#dataframe = pandas.read_csv("feature.csv", header=None)
dataframe = pandas.read_csv("input_glat2.csv", header=None)
dataset = dataframe.values
np.random.shuffle(dataset[1:])
print(dataset)
# split into input (X) and output (Y) variables
#X = dataset[1:1933,0:5].astype(float)
X = dataset[1:1933,0:6].astype(float)
#Y = dataset[1:1933,5]
Y = dataset[1:1933,6]
num=[]
num2=[]
resul=[]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
i=60
print(X)
print(encoded_Y)
weight1=800/166
weight2=800/1739
class_weights= {1: weight1, 0: weight2}
# baseline model
j=10
def create_baseline():
	# create model
	model = Sequential()
	#model.add(Dense(3, input_dim=5, kernel_initializer='normal', activation='tanh'))
	model.add(Dense(3, input_dim=6, kernel_initializer='normal', activation='tanh'))
	model.add(Dense(1,kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# evaluate baseline model with standardized dataset

while i == 60:
        while j == 10 :
                estimators = []
                estimators.append(('standardize', StandardScaler()))
                estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=i, batch_size=j, verbose=0)))
                pipeline = Pipeline(estimators)
                kfold = StratifiedKFold(n_splits=20, shuffle=True, random_state=seed)
                results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
                print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
                num.append(i)
                num2.append(j)
                j=j+5
                print(i)
                print(j)
                resul.append(results.mean()*100)
        i=i+50
        j=10

fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(num, num2, resul, 'o')
ax.set_xlabel('epoch')
ax.set_ylabel('batch_size')
ax.set_zlabel('score')
ax.set_title('epoch vs. batch_size vs. score for (3,1) architecture with tanh and sigmoid')
plt.show()
        
'''

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=200, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



fig,ax=plt.subplots()
plt.plot(mod.history['acc'])
plt.plot(mod.history['val_acc'])
ax.set_xlabel('epoch')
ax.set_ylabel('Accuracy')
ax.legend(['train','test'],loc='upper left')
plt.show()
#plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')

#his=mod.history["loss"]
#plt.plot(his)
model.summary()
fig2,ax2=plt.subplots()
#print(mod.history['val_loss'])
plt.plot(mod.history['loss'])
plt.plot(mod.history['val_loss'])
ax2.set_xlabel('epoch')
ax2.set_ylabel('loss')
ax2.legend(['train','test'],loc='upper left')
plt.show()
'''
