import numpy as np
import matplotlib.pyplot as plt
from random import shuffle, sample
import pandas as pd
import seaborn as sns


plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 25
plt.rcParams['font.size'] = 14
plt.rcParams['ytick.labelsize'] = 14


seed=5
np.random.seed(seed)
'''
file_1=fits.open('input3')
file_2=fits.open('OUTPUT')
file_3=fits.open('input_glat')
'''
dataframe = pd.read_csv("./files/4fgldr2_all.csv", header=None)
dataset=dataframe.values
print(dataset[0,38])
labels2=dataset[0:1,1:29]
labels2=labels2.ravel()
#dataset=dataset[1:,1:29].astype(float)
dataset=[dataset[i,1:29].astype(float) for i in range(len(dataset)) if dataset[i,38]=='AGN' or dataset[i,38]=='PSR']# or dataset[i,12]=='OTHER']

dataset=pd.DataFrame(dataset,columns=labels2)
#plt.rc('xtick', labelsize=7) 
#plt.rc('ytick', labelsize=10)
corr=dataset.corr(method='pearson')
mask = np.triu(np.ones_like(corr, dtype=np.bool))
corr2=abs(corr)
#mask2=np.less(corr2,0.7)
#mask=mask +mask2
#corr=corr*-1
print(mask)
#print(sai)
fig, ax = plt.subplots(figsize=(31,31))         # Sample figsize in inches
ax.set_title("Correlation in 3FGL Associated Data")

sns.heatmap(corr, mask=mask, annot=True,annot_kws={"size": 8},cbar_kws={"shrink": .5},linewidths=3)
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
