# auxiliary functions for the FGL-ML project

import numpy as np
import pandas as pd

def tform2dtype(tform):
    if tform.find('A') > -1:
        return str
    elif tform.find('E') > -1 or tform.find('D') > -1:
        if len(tform) == 1:
            return float
        else:
            return 'array'
    elif tform.find('I') > -1:
        if len(tform) == 1:
            return int
        else:
            return 'array'

def hdu2df(table, index_name=None):
    if index_name is not None:
        index = np.array(table.data.field(index_name), dtype=str)
        index = [st.strip() for st in index]
    all_keys = list(table.header.keys())
    data = {}
    for key in all_keys:
        if key.startswith('TFORM'):
            form = table.header[key]
            dtype = tform2dtype(form)
            if dtype != 'array':
                type_key = key.replace('TFORM', 'TTYPE')
                data_key = table.header[type_key]
                if data_key != index_name:
                    data[data_key] = np.array(table.data.field(data_key), dtype=dtype)
    return pd.DataFrame(data=data, index=index)
    #df_fgl.index = [st.strip() for st in df_fgl.index]

    
def get_prob_class(df, algs, classes):
    res = pd.DataFrame(index=df.index)
    res['Category_Prob'] = 'MIXED'
    masks = {}
    classes_loc = [cls for cls in classes if ('%s_%s' % (cls, algs[0])) in df.columns]
    for cls in classes_loc:
        masks[cls] = 1.
    for alg in algs:
        #print(alg)
        columns = ['%s_%s' % (cls, alg) for cls in classes_loc]
        thres = np.max(df[columns], axis=1) - 1.e-5
        #print(thres)
        for cls in classes_loc:
            clm = '%s_%s' % (cls, alg)
            masks[cls] *= np.heaviside(df[clm] - thres, 0.)
            #print(cls, masks[cls])
    for cls in classes_loc:
        msk = np.array(masks[cls], dtype=bool)
        res['Category_Prob'][msk] = cls
    return res

def accuracy(true_labels, pred_labels):
    return np.sum(true_labels == pred_labels) / len(true_labels)


def h2cum(arr):
    res = 1. * arr[::-1]
    for i in range(1, len(arr)):
        res[i] += res[i - 1]
    return res[::-1]

def min_max_vs(dct, corr={}, keys=None):
    if keys is None:
        keys = dct.keys()
    vals = np.array([dct[key] - corr.get(key, 0.) for key in keys])
    return np.min(vals, axis=0), np.max(vals, axis=0)

def get_mean_dp_dm(pred):
    mean = np.mean(pred)
    delta_plus = np.max(pred) - mean
    delta_minus = mean - np.min(pred)
    return mean, delta_plus, delta_minus
