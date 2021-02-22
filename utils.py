from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.multiclass import _ConstantPredictor, OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pandas as pd
from collections import Counter
import random
import os
import time
import gc


# import torch

def seed_everything(seed=42):
    # seed the random number generators so that we may reproduce the results
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


def _fit_binary_new(estimator, X, y, sample_weight, classes=None):
    # used by OneVsRestClassifierNew
    unique_y = np.unique(y)
    if len(unique_y) == 1:
        if classes is not None:
            if y[0] == -1:
                c = 0
            else:
                c = y[0]
            warnings.warn("Label %s is present in all training examples." %
                          str(classes[c]))
        estimator = _ConstantPredictor().fit(X, unique_y)
    else:
        estimator = clone(estimator)

        # Only this changed
        estimator.fit(X, y, sample_weight=sample_weight)
    return estimator


class OneVsRestClassifierNew(OneVsRestClassifier):
    # a class based on sklearn's OneVsRestClassifier that allows specifying a weight for each sample

    def fit(self, X, y, sample_weight=None):
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(delayed(_fit_binary_new)(
            self.estimator, X, column, sample_weight, classes=[
                "not %s" % self.label_binarizer_.classes_[i],
                self.label_binarizer_.classes_[i]])
                                                        for i, column in enumerate(columns))

        return self


def get_train_df():

    # load the pre-processed data and add a "target2" column that buckets all the classes that are not normal, class10 and class18

    t0 = time.time()
    train = pd.read_pickle("./files/train2.pkl")
    t1 = time.time()
    print(f"... total elapsed = {t1 - t0}")
    print(train.shape)

    target = train["target"]

    t0 = time.time()
    a = target.values
    target2 = [x if x in ["normal", "class10", "class18"] else 'other' for x in a]
    t1 = time.time()
    print(f"... total elapsed = {t1 - t0}")

    del a
    gc.collect()

    df_target2 = pd.DataFrame(data=target2, columns=["target2"])
    train = pd.concat([train, df_target2], axis=1)

    return train
