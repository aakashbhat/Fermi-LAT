import os
import pandas
import numpy as np
directory = 'D:/work_projects/fermilat/Fermi-LAT/ML_3FGL_paper/data/'

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        print(filename)
        df = pandas.read_csv(filename, header=None)
        print(df)
        findL1 = ['Other','CLASS2']
        replaceL1=['OTHER','Category_4FGL']
        coll=0
        k=np.shape(df)
        print(k[1])
        for i in range(k[1]):
            df2=df[i]
            print(df2[0])
            if df2[0]=="3FGL":
                coll=i
        print(df[coll])
        df[coll] = '3FGL '+df[coll]

        print(df[coll])

        df.to_csv(path_or_buf="D:/work_projects/fermilat/Fermi-LAT/ML_3FGL_paper/data/sazparkinson2.csv".format(a=filename),index=False)

