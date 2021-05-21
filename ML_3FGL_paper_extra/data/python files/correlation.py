import numpy as np
import matplotlib.pyplot as plt
from random import shuffle, sample
import pandas as pd
import seaborn as sns
import sklearn.feature_selection
from sklearn.metrics import mutual_info_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 25
plt.rcParams['font.size'] = 18
plt.rcParams['ytick.labelsize'] = 18


seed=5
np.random.seed(seed)
'''
file_1=fits.open('input3')
file_2=fits.open('OUTPUT')
file_3=fits.open('input_glat')
'''
dataframe = pd.read_csv("./files/3fglassoc.csv", header=None)
dataset=dataframe.values
#print(dataset)
dataset2=dataframe.values[0]
labels2=dataset2[0:20]
#print(labels2)
labels2=labels2.ravel()
#print(dataset[0,17])
#labels2=dataset[0:1,1:12]
#labels2=labels2.ravel()
#dataset=dataset[1:,1:29].astype(float)
#print(dataset)
mat=np.zeros((20,20))
'''
for j in range(20):
    for k in range(20):
        X=[dataset[i,j+1] for i in range(len(dataset)) if dataset[i,21]=='AGN' or dataset[i,21]=='PSR']# or dataset[i,12]=='OTHER']
        Y=[dataset[i,k+1] for i in range(len(dataset)) if dataset[i,21]=='AGN' or dataset[i,21]=='PSR']# or dataset[i,12]=='OTHER']
        X=np.asarray(X)
        Y=np.asarray(Y)
        X=X.astype(float)
        Y=Y.astype(float)
        c_xy = np.histogram2d(X, Y, 30)[0]
        mi = mutual_info_score(None,None,contingency=c_xy)
        print(dataset2[j+1],dataset2[k+1],mi)
        mat[j,k]=mi
'''

dataset=dataset[1:,0:20].astype(float)
df=pd.DataFrame(dataset,columns=labels2)
#print(df)
#plt.rc('xtick', labelsize=7) 
#plt.rc('ytick', labelsize=10)
#corr=sklearn.feature_selection.mutual_info_classif(X,Y)
corr1=df.corr(method='pearson')
print(corr1)
mask = np.triu(np.ones_like(corr1, dtype=np.bool))
#corr2=abs(corr1)
#mask2=np.less(corr2,0.7)
#mask=mask +mask2
fig, ax = plt.subplots(figsize=(31,31))         # Sample figsize in inches
ax.set_title("Correlation in 3FGL Associated Data")
sns.heatmap(corr1, mask=mask, annot=True,annot_kws={"size": 8},cbar_kws={"shrink": .5},linewidths=3)
plt.xticks(rotation='vertical')
plt.yticks(rotation='horizontal')

plt.show()



#mat = StandardScaler().fit_transform(mat)
#print(mat)
mask = np.triu(np.ones_like(corr, dtype=np.bool))
corr2=abs(corr)
#mask2=np.less(corr2,0.7)
#mask=mask +mask2
#corr=corr*-1
#print(mask)
#print(sai)






fig, ax = plt.subplots(figsize=(31,31))         # Sample figsize in inches
ax.set_title("Mutual Information in 3FGL Associated Data")
mat3=pd.DataFrame(mat,columns=labels2)
for column in mat3.columns: 
    mat3[column] = mat3[column]  / mat3[column].abs().max() 
#min_max_scaler = preprocessing.MinMaxScaler()
#np_scaled = min_max_scaler.fit_transform(mat3)
scaler = StandardScaler()
#mat3 = pd.DataFrame(np_scaled)
sns.heatmap(mat3,xticklabels=labels2, yticklabels=labels2, annot=True,annot_kws={"size": 8},cbar_kws={"shrink": .5},linewidths=3)
plt.show()


# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(33, 31))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
ax.set_title("Highly co-related 3FGL Associated Data")
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap,center=0,annot=True,annot_kws={"size": 8}, linewidths=3, cbar_kws={"shrink": .5})
plt.show()
def heatmap(x, y, size):
    fig, ax = plt.subplots()
    
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    size_scale = 500
    ax.scatter(
        x=x.map(x_to_num), # Use mapping for x
        y=y.map(y_to_num), # Use mapping for y
        s=size * size_scale, # Vector of square sizes, proportional to size parameter
        marker='s' # Use square as scatterplot marker,
        
    )
 
    #sns.palplot(sns.diverging_palette(220, 20, n=7))

    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5]) 
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    plt.show()
  
data = pd.read_csv("4fgl_assoc_1.csv", header=None)
data=data.values
#print(labels2.shape())
columns=labels2 
print(columns)
corr = data[labels2].corr()
#corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
#corr.columns = ['x', 'y', 'value']
'''
heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs()
)
'''
#corrmat = data.corr() 
  
f, ax = plt.subplots(figsize =(32, 32)) 
sns.heatmap(corr, ax = ax, cmap ="YlGnBu", linewidths = 0.1)
plt.show()
