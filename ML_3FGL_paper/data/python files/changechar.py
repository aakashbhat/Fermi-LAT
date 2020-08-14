import os
import pandas
import numpy as np
directory = 'D:/work_projects/fermilat/Fermi-LAT/ML_3FGL_paper/data/'

df = pandas.read_csv("sazparkinson.csv", header=None)
print(df)
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        print(filename)
        df = pandas.read_csv(filename, header=None)
        #findL1 = ['Other','CLASS2']
        #replaceL1=['OTHER','Category_4FGL']
        coll=0
        k=np.shape(df)
        print(k[1])
        for i in range(k[1]):
            df2=df[i]
            print(df2[0])
            if df2[0]=="Category_3FGL" or df2[0]=="CLASS2":
                coll=i
        print(df[coll])
        df[coll] = df[coll].replace(findL1, replaceL1)

        print(df[11])

        df.to_csv(path_or_buf="D:/work_projects/fermilat/Fermi-LAT/ML_3FGL_paper/data/catalogs/catalogs2/{a}".format(a=filename),index=False)

