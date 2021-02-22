import pandas as pd
import numpy as np
from collections import Counter
import pickle
from utils import get_train_df, OneVsRestClassifierNew, seed_everything
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
import gc
import time
from scipy.optimize import minimize


def prob_array_to_class(X_in, factor_vec, classes):  # convert prediction matrix into class prediction vector, given a factor vector
    X = X_in.copy()
    n = -1
    for f in factor_vec:
        n += 1
        X[:, n] = X[:, n] * f  # multiply each column by its corresponding factor

    ind = np.argmax(X, axis=1)  # predict by the index of the maximum
    y_pred = classes[ind]  # convert to class name
    return y_pred


def optimize_factors(X, y_test_classes, classes):  # optimize the factor vector to maximize the balanced accuracy
    x0 = np.ones(len(classes) - 1)
    t0 = time.time()
    res = minimize(ba_func, x0, args=(X, y_test_classes, classes), method='powell')  # we sklearn has minimizing procedures. therefore we minimize -1*ba_func
    t1 = time.time()
    print(f"optimization took {t1 - t0}")

    f_vec = res.x
    f_vec = np.append(f_vec, 1)
    return f_vec


def ba_func(f_vec, X, y_test_classes, classes):
    f_vec = np.append(f_vec, 1)
    y_pred_all_classes = prob_array_to_class(X, f_vec, classes)  # get class predictions
    ba = balanced_accuracy_score(y_test_classes, y_pred_all_classes)  # calculate balanced accuracy
    return -ba  # we multiply by -1 since we minimize and not maximize


if __name__ == "__main__":

    seed_everything()  # seeding the random number generators so we could reproduce the results

    n_folds = 100
    n_trees = 50

    train = get_train_df()  # get the pre-processed train data

    cols = train.columns

    target = train['target']
    target2 = train['target2']

    hist_target2 = Counter(target2)

    train_other = train.loc[target2 == 'other']
    target_other = target.loc[target2 == 'other']
    del train
    gc.collect()

    hist_target = Counter(target_other)
    print(train_other.shape)

    large_classes = [k for k in hist_target if hist_target[k] > 950]  # we classified the larger classes plus class07 and class14 in the previous phases
    large_classes.append('class07')
    large_classes.append('class14')

    target_other2 = np.array([x if x in large_classes else 'other2' for x in target_other])  # we find the location ot the smaller classes (given as "other2")

    train_other3 = train_other.loc[target_other2 == 'other2']  # we keep the data only of the smaller classes
    target_other3 = target_other.loc[target_other2 == 'other2']

    hist_target3 = Counter(target_other3)

    classes = np.unique(target_other3)  # the smaller classes
    ind_classes = {}
    models_per_class = {}
    for c in classes:  # for each class we collect its indices
        ind = np.where(target_other3 == c)[0]
        ind_classes[c] = ind

    # x = pd.DataFrame(target_other3)
    # x.columns = ['target3']

    pred_all = np.zeros((train_other3.shape[0], len(classes)))

    y_pred_all = None
    y_test_all = None
    y_valid_all = None
    y_valid_test_all = None
    classes_in_clf0 = None

    models = []
    err_rate_array = []

    for f in range(n_folds):

        print(f"fold={f}")

        ind_train = []
        ind_test = []
        for c in classes:
            ind = target_other3.index[target_other3 == c]
            perm = np.random.permutation(len(ind)).tolist()
            N = int(len(ind) * 2 / 3)  # for each class ~2/3 of the samples are for train. the rest are for test
            ind1 = list(ind[perm[:N]])
            ind2 = list(ind[perm[N:]])
            ind_train = ind_train + ind1
            ind_test = ind_test + ind2

        assert (len(np.intersect1d(ind_train, ind_test)) == 0)  # check we do not mix train and test
        assert (len(target_other3) == len(ind_train) + len(ind_test))  # check we exhaust all of the samples

        X_train = train_other3.loc[ind_train]  # this is the train matrix
        z = np.unique(X_train.target)
        assert (len(z) == len(classes))
        y_train = X_train.target  # the target function (train)

        X_test = train_other3.loc[ind_test]  # this is the test matrix
        z = np.unique(X_test.target)
        assert (len(z) == len(classes))
        y_test = X_test.target  # the target function (test)

        X_train = X_train.drop(columns=['target', 'target2'])
        X_test = X_test.drop(columns=['target', 'target2'])

        weights = np.zeros(y_train.shape)  # calculate sample weights
        hist_y_train = Counter(y_train)
        n = 0
        for c in y_train:
            weights[n] = 1 / hist_y_train[c]  # the weights per row are inversely proportional to the amount of the class in the data
            n += 1

        clf_in = RandomForestClassifier(verbose=0, n_estimators=n_trees, random_state=0)  # we use a RF model as the inner classifier type in OneVsRest
        clf = OneVsRestClassifierNew(clf_in)  # we use OneVsRestClassifierNew and not OneVsRestClassifier so that we could use the sample weights
        clf.fit(X_train, y_train, sample_weight=weights)  # here we use the sample weights.

        # we check that the classes in the models are arrange the same for each iteration
        classes_in_clf = clf.classes_
        if classes_in_clf0 is None:
            classes_in_clf0 = classes_in_clf

        for (c, c2) in zip(classes_in_clf0, classes_in_clf):
            assert (c == c2)

        y_pred = clf.predict_proba(X_test)  # predict the test

        if f <= n_folds / 2:
            models.append(clf)  # we keep the models for the first half of folds
            if y_pred_all is None:
                y_pred_all = y_pred
            else:
                y_pred_all = np.append(y_pred_all, y_pred, axis=0)
            if y_test_all is None:  # we save the test and predicted data
                y_test_all = y_test
            else:
                y_test_all = np.append(y_test_all, y_test, axis=0)
        else:

            if y_valid_all is None:  # we use the second part of the folds as validation
                y_valid_all = y_pred
            else:
                y_valid_all = np.append(y_valid_all, y_pred, axis=0)
            if y_valid_test_all is None:
                y_valid_test_all = y_test
            else:
                y_valid_test_all = np.append(y_valid_test_all, y_test, axis=0)

    f_vec = np.ones(len(classes_in_clf0))
    y_pred_all_classes = prob_array_to_class(y_pred_all, f_vec, classes_in_clf0)  # predict with factor vec of ones

    ba = balanced_accuracy_score(y_test_all, y_pred_all_classes)  # initial balanced accuracy on the test data
    print(f"balanced accuracy test={ba}")

    err_classes = {}
    for c in classes_in_clf0:
        err_classes[c] = set()
    for (c, c2) in zip(y_test_all, y_pred_all_classes):
        if c != c2:
            err_classes[c].add(c2)

    # ---------------------------------------------
    f_vec_optimized = optimize_factors(y_pred_all, y_test_all, classes_in_clf0)  # optimize the factor vec

    # validate
    y_valid_all_classes = prob_array_to_class(y_valid_all, f_vec, classes_in_clf0)  # test the improvement in balanced accuracy on the validation data
    ba_valid_before = balanced_accuracy_score(y_valid_test_all, y_valid_all_classes)
    print(f"balanced accuracy validation={ba_valid_before}")
    y_valid_all_classes = prob_array_to_class(y_valid_all, f_vec_optimized, classes_in_clf0)
    ba_valid_after = balanced_accuracy_score(y_valid_test_all, y_valid_all_classes)
    print(f"balanced accuracy valid with optimized factors={ba_valid_after}")

    print(f"f_vec_optimized={f_vec_optimized}")

    a = 1

    pickle.dump(models, open("./files/third_phase_models.pkl", 'wb'))  # save the models
    pickle.dump(f_vec_optimized, open("./files/third_phase_models_f_vec_optimized.pkl", 'wb'))  # save the vector of factors
