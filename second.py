# from first import get_train_df, seed_everything
# from utils import get_train_df
import pandas as pd
import numpy as np
import time
from collections import Counter
import pickle
# from sklearn.multiclass import OneVsRestClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
# from sklearn.metrics import balanced_accuracy_score
# import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from utils import get_train_df, OneVsRestClassifierNew, seed_everything


# from optimize_thresholds import optimize_thresholds_D,get_err_count

def get_CV_folds(train, target, n_folds):
    x = pd.DataFrame(target)
    x.columns = ['target2']
    target_one_hot = pd.get_dummies(x, columns=['target2'])

    mskf = MultilabelStratifiedKFold(n_splits=n_folds)

    folds_out = {}
    for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target_one_hot)):
        folds_out[f] = v_idx

    return folds_out


if __name__ == "__main__":

    seed_everything()  # seeding the random number generators so we could reproduce the results

    n_folds = 8
    n_trees = 100

    train = get_train_df()  # get the pre-processed train data
    cols = train.columns

    target = train['target']  # original classes
    target2 = train['target2']  # classes from phase 1 (4 classes)

    hist_target1 = Counter(target)
    hist_target2 = Counter(target2)

    train_other = train.loc[target2 == 'other']  # in the second phase we are working only on the rows of the "other" class
    target_other = target.loc[target2 == 'other']

    hist_target = Counter(target_other)  # count the amount of samples of each class
    print(train_other.shape)

    medium_classes = [k for k in hist_target if hist_target[k] > 950]  # in this phase we are working on the medium sized classes
    medium_classes.append('class07')  # we add class07 and class14 also, since they were proved to be perfectly predicted (hence we bounced them
    medium_classes.append('class14')  # up from the 3rd phase to the 2nd phase).

    target_other2 = np.array([x if x in medium_classes else 'other2' for x in target_other])  # the "smaller classes" are now put into the "other2" bucket.
    hist_target2 = Counter(target_other2)

    classes = np.unique(target_other2)  # here are the classes we try to classify in this phase

    # split into folds
    train_other2 = train_other.drop(columns=['target', 'target2'])  # drop columns "target" and "target2" from the data
    folds = get_CV_folds(train_other2, target_other2, n_folds)
    # prepare prediction array on the test set: (it is the same size as the complete train data. in each fold we leave a "chunk" out. train on the rest and predict the "chunk")
    pred_all = np.zeros((train_other.shape[0], len(classes)))
    # iterate the cv folds:
    models = []
    for f in folds:

        ind_train = []
        ind_test = list(folds[f])  # folds[f] contains the test data indices for the fold (the chunk we leave out)

        for f2 in folds:
            if f == f2:
                continue
            ind_train_temp = list(folds[f2])
            ind_train = ind_train + ind_train_temp  # we aggregate the rest of the indices as train data

        assert (len(ind_train) + len(ind_test) == len(target_other2))  # check we exhaust all of the data

        # create Train and Test matrices
        X_train = train_other.iloc[ind_train]
        Y_train = target_other2[ind_train]
        Y_train_classes_orig = X_train['target']  # original classes: will be used to give weight for the samples

        X_test = train_other.iloc[ind_test]
        Y_test = target_other2[ind_test]

        X_train = X_train.drop(columns=['target', 'target2'])
        X_test = X_test.drop(columns=['target', 'target2'])

        print(f"fold #{f}: just before OneVsRestClassifier")
        t0 = time.time()

        weights = np.zeros(Y_train_classes_orig.shape)  # prepare vector of sample weights
        hist_y_train = Counter(Y_train_classes_orig)
        n = 0
        for c in Y_train_classes_orig:
            weights[n] = 1 / hist_y_train[c]  # the weights per row are inversely proportional to the amount of the class in the data
            n += 1

        clf_in = RandomForestClassifier(verbose=0, n_estimators=n_trees, random_state=0)  # we use a RF model as the inner classifier type in OneVsRest
        clf = OneVsRestClassifierNew(clf_in)  # we use OneVsRestClassifierNew and not OneVsRestClassifier so that we could use the sample weights
        clf.fit(X_train, Y_train, sample_weight=weights)  # here we use the sample weights.

        t1 = time.time()
        print(f"fold #{f}: OneVsRestClassifier elapsed = {t1 - t0}")
        pred_all[ind_test] = clf.predict_proba(X_test)  # predict the test data

        models.append(clf)

    # check that the models are all arranged the same way:
    classes = models[0].classes_
    for m in models:
        for i in range(len(classes)):
            assert (classes[i] == m.classes_[i])

    # init:
    y_pred_all = np.argmax(pred_all, axis=1)
    y_pred_classes = np.array(['abbaaaaaaaaaaaa' for _ in range(target_other2.shape[0])])
    err_count = {}
    total_count = {}
    n = -1
    for c in classes:
        n += 1
        y_pred_classes[y_pred_all == n] = c
        err_count[c] = 0
        total_count[c] = 0

    for (c, c2) in zip(target_other2, y_pred_classes):
        total_count[c] += 1
        if c != c2:
            err_count[c] += 1

    # here we calculate the factor to multiply the "other2" class prediction so as to maximize the balanced accuracy (giving a higher weight to the "other2" class)
    err_count0 = err_count

    f_vec = np.logspace(-2, 6, num=101)  # possible factors are log spaced
    N = pred_all.shape[1] - 1
    ba_max = -np.inf
    factor_max = -1
    for factor in f_vec:  # we iterate over possible factors (log spaced)
        pred_all_temp = pred_all.copy()

        pred_all_temp[:, N] = pred_all_temp[:, N] * factor  # multiply the last column by a factor

        y_pred_all = np.argmax(pred_all_temp, axis=1)  # index of predicted class
        y_pred_classes = np.array(['abbaaaaaaaaaaaa' for _ in range(target_other2.shape[0])])
        err_count = {}
        total_count = {}
        n = -1
        for c in classes:
            n += 1
            y_pred_classes[y_pred_all == n] = c  # convert to class name
            err_count[c] = 0
            total_count[c] = 0

        for (c, c2) in zip(target_other2, y_pred_classes):  # calculate errors
            total_count[c] += 1
            if c != c2:
                err_count[c] += 1

        ba = 0  # calculate balanced accuracy with higher weight for "other2" class
        for c in classes:
            if c == "other2":
                ba = ba + (1 - err_count[c] / total_count[c] * 11)  # higher weight to the "other2" class
            else:
                ba = ba + (1 - err_count[c] / total_count[c])
        ba = ba / (len(classes) + 10)
        print(err_count)
        print(ba)
        if ba > ba_max:  # if we get higher balanced accuracy then we save the factor used.
            factor_max = factor
            ba_max = ba
            err_count1 = err_count

    print(err_count0)
    print(err_count1)

    err_rate0 = {}
    err_rate1 = {}
    for c in classes:
        err_rate0[c] = err_count0[c] / total_count[c]
        err_rate1[c] = err_count1[c] / total_count[c]

    print(f"err_rate0={err_rate0}")
    print(f"err_rate1={err_rate1}")
    print(f"factor_max={factor_max}")

    pickle.dump(models, open("./files/second_phase_models.pkl", 'wb'))  # save models
    pickle.dump(factor_max, open("./files/second_phase_models_factor_max.pkl", 'wb'))  # save factor
