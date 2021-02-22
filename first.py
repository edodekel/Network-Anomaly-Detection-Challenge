import pandas as pd
import numpy as np
from collections import Counter
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
import pickle
from optimize_thresholds import optimize_thresholds_D
from utils import get_train_df, OneVsRestClassifierNew, seed_everything

if __name__ == "__main__":

    seed_everything()  # seeding the random number generators so we could reproduce the results

    folds = 4
    n_trees = 100

    train = get_train_df()  # get the pre-processed train data
    hist_target = Counter(train['target'])  # check the amount of samples of each class

    cols = train.columns
    target2 = train['target2']
    hist_target2 = Counter(target2)

    train_normal = train.loc[target2 == 'normal']  # divide the data into 4 classes
    train_class10 = train.loc[target2 == 'class10']
    train_class18 = train.loc[target2 == 'class18']
    train_other = train.loc[target2 == 'other']

    s_normal = train_normal.shape[0]
    s_class10 = train_class10.shape[0]
    s_class18 = train_class18.shape[0]
    s_other = train_other.shape[0]

    total_rows = s_normal + s_class10 + s_class18 + s_other

    # sanity:
    assert (total_rows == train.shape[0])

    N = 100000  # there are "too" many samples of "normal", "class10" and "class18". in each fold we select random 100k for training
    N_other = 30000  # and 30k samples from the "other" class

    N_other_test = hist_target2['other'] - N_other  # test samples are the rest of the samples of "other"
    N_normal_test = int(hist_target2['normal'] / hist_target2['other'] * N_other_test)  # and random by the original amount ratio for "normal", "class10" and "class18"
    N_class10_test = int(hist_target2['class10'] / hist_target2['other'] * N_other_test)
    N_class18_test = int(hist_target2['class18'] / hist_target2['other'] * N_other_test)

    th_vec = []
    models = []
    for f in range(folds):

        print(f"-------------------")
        print(f"FOLD {f}")
        print(f"-------------------")

        perm = np.random.permutation(s_normal)  # for each class randomly divide into train/test indices
        ind_normal_train = perm[:N]
        ind_normal_test = perm[N:N + N_normal_test]

        perm = np.random.permutation(s_class10)
        ind_class10_train = perm[:N]
        ind_class10_test = perm[N:N + N_class10_test]

        perm = np.random.permutation(s_class18)
        ind_class18_train = perm[:N]
        ind_class18_test = perm[N:N_class18_test]

        perm = np.random.permutation(s_other)
        ind_other_train = perm[:N_other]
        ind_other_test = perm[N_other:]

        # create train matrix
        Xy = pd.concat([train_normal.iloc[ind_normal_train], train_class10.iloc[ind_class10_train], train_class18.iloc[ind_class18_train], train_other.iloc[ind_other_train]], axis=0)
        print(Xy.shape)

        # create test matrix
        Xy_test = pd.concat([train_normal.iloc[ind_normal_test], train_class10.iloc[ind_class10_test], train_class18.iloc[ind_class18_test], train_other.iloc[ind_other_test]], axis=0)
        print(Xy_test.shape)

        # divide into data (X) and target (y)
        y_train_classes_orig = Xy['target']  # we use the original classes to give weight for each sample
        y_train_classes = Xy['target2']
        X_train = Xy.drop(columns=['target', 'target2'])

        y_test_classes = Xy_test['target2']
        X_test = Xy_test.drop(columns=['target', 'target2'])

        # now, build models for this fold:
        print(f"fold #{f}: just before OneVsRestClassifier")
        t0 = time.time()

        weights = np.zeros(y_train_classes_orig.shape)
        hist_y_train = Counter(y_train_classes_orig)
        n = 0
        for c in y_train_classes_orig:
            weights[n] = 1 / hist_y_train[c]  # each data row has it's own weight according inversely proportional to the amount of examples of the class
            n += 1

        clf_in = RandomForestClassifier(verbose=0, n_estimators=n_trees, random_state=0)  # we use a RF model as the inner classifier type in OneVsRest
        clf = OneVsRestClassifierNew(clf_in)  # we use OneVsRestClassifierNew and not OneVsRestClassifier so that we could use the sample weights
        clf.fit(X_train, y_train_classes, sample_weight=weights)  # here we use the sample weights.
        t1 = time.time()
        print(f"fold #{f}: OneVsRestClassifier elapsed = {t1 - t0}")

        models.append(clf)  # we append the models. we will later use all of them

        # collect some statistics on error rate

        pred_all = clf.predict_proba(X_test)  # predict the classes of the test data

        y_pred_all = np.argmax(pred_all, axis=1)  # here we use a simple maximum to select the correct class
        y_pred_classes = np.array(['abbaaaaaaaaaaaa' for _ in range(y_test_classes.shape[0])])  # prepare "empty" array of strings
        y_pred_classes_num = np.zeros(y_test_classes.shape)
        y_test_classes_num = np.zeros(y_test_classes.shape)

        err_count = {}
        total_count = {}
        classes = clf.classes_
        n = -1
        for c in classes:
            n += 1
            y_pred_classes[y_pred_all == n] = c
            err_count[c] = 0  # prepare the err_count and total_count dictionaries
            total_count[c] = 0
        n = -1
        for c in classes:
            n += 1
            y_pred_classes_num[y_pred_classes == c] = n
            y_test_classes_num[y_test_classes == c] = n  # the index of the classes. used later when optimizing the thresholds

        for (c, c2) in zip(y_test_classes, y_pred_classes):
            total_count[c] += 1
            if c != c2:
                err_count[c] += 1

        print(err_count)
        ba = balanced_accuracy_score(y_test_classes, y_pred_classes)
        th = optimize_thresholds_D(pred_all, y_test_classes_num)  # get a factor for the "other" class to increase the probability it is selected (to improve balanced accuracy)
        th_vec.append(th)
        print(th)

    pickle.dump(models, open("./files/first_phase_models_all.pkl", 'wb'))  # save the models and the factors for the prediction stage
    pickle.dump(th_vec, open("./files/first_phase_th_vec.pkl", 'wb'))
