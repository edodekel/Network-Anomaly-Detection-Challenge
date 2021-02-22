import pandas as pd
import numpy as np
import datatable as dt
import time
import gc
import sys

# test if pre-processed data files already exists

try:
    f = open("./files/train2.pkl")
    f.close()


except: # pre-processed data does not exist. we create it

    print("pre-processing")
    # load data from csv:
    t0 = time.time()
    train = pd.read_csv('files/training.csv', header=None)
    t1 = time.time()
    print(f"... total elapsed = {t1 - t0}")
    print(train.shape)

    t0 = time.time()
    test = pd.read_csv('files/eval-rev2.csv', header=None)
    t1 = time.time()
    print(f"... total elapsed = {t1 - t0}")
    print(test.shape)

    # add test target column
    test['target']='?'

    # rename columns:
    cols = train.columns
    cols = ["feature" + "_" + str(x) for x in cols]
    cols[len(cols) - 1] = "target"

    train.columns=cols
    test.columns=cols

    # concatenate train and test data:
    Xy=pd.concat([train,test],axis=0)
    print(Xy.shape)

    del train
    del test
    gc.collect()

    # count number of unique values per column
    unique_vals = {}
    num_unique_vals = {}
    for col in cols:
        col_vals = Xy[col]
        unique_vals[col] = np.unique(col_vals)
        num_unique_vals[col] = len(unique_vals[col])
        a = 1

    # reduce memory (int64->int32, float64->float32):
    print(sys.getsizeof(Xy))
    col_type = Xy.dtypes
    c = -1
    for col in cols:
        c += 1

        if col_type[c] == np.dtype('int64'):
            if max(unique_vals[col] < 2 ^ 31):
                Xy[col] = Xy[col].astype('int32')
            else:
                print('integers larger then int32')
        if col_type[c] == np.dtype('float64'):
            Xy[col] = Xy[col].astype('float32')
    print(sys.getsizeof(Xy))

    # check which columns to convert to one-hot columns:
    one_hot_cols = []
    for col in cols:
        if col.startswith("feature_"):
            if col_type[col] == np.dtype('O'):  # categorical columns
                one_hot_cols.append(col)
            else:
                if num_unique_vals[col] > 2 and num_unique_vals[col] < 10:  # columns with less then 10 distinct values
                    one_hot_cols.append(col)


    Xy=pd.get_dummies(Xy,columns=one_hot_cols)

    print(Xy.shape)

    # split back to train/test
    test=Xy[Xy.target=='?']
    train=Xy[Xy.target!='?']
    print(train.shape)
    print(test.shape)

    train.to_pickle("./files/train2.pkl")
    test.to_pickle("./files/test2.pkl")
