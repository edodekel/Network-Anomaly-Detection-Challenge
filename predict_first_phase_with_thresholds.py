import pandas as pd
import numpy as np
import time
from collections import Counter
import pickle

t0 = time.time()
test = pd.read_pickle("./files/test2.pkl")
t1 = time.time()
print(f"... total elapsed = {t1 - t0}")
test = test.drop(columns=['target'])

# first_phase:

with (open('./files/first_phase_models_all.pkl', "rb")) as openfile:  # read models
    while True:
        try:
            models = pickle.load(openfile)
        except EOFError:
            break

with (open('./files/first_phase_th_vec.pkl', "rb")) as openfile:  # read factor vec
    while True:
        try:
            factor_vec = pickle.load(openfile)
        except EOFError:
            break

factor = np.max(factor_vec) * 1.25  # we use the maximum of all factors seen and we add 25 percent to be on safe side.
# factor = np.median(factor_vec) * 4  # we add 300% percent to be on safe side. and use median for robustness
# factor = 200

classes = models[0].classes_
pred_all = np.zeros((test.shape[0], len(classes)))

c = -1
models_types_ind = {}
models_types_ind2 = {}

L = len(classes) - 1
assert (models[0].classes_[L] == "other")
fold = -1
for model in models:  # sum the predicted probability of each model
    fold += 1
    print(f"fold #{fold}")
    y_pred = model.predict_proba(test)
    pred_all += y_pred
    for i in range(L + 1):
        assert (model.classes_[i] == classes[i])  # we recheck that all the models are arranged the same

pred_all[:, L] = pred_all[:, L] * factor  # increase the weight of the "other" class

pred_all_ind = np.argmax(pred_all, axis=1)  # predict class
pred_all_classes = np.array(['abbaaaaaaaaaaaa' for _ in range(len(pred_all_ind))])
c = -1
for ind in pred_all_ind:
    c += 1
    pred_all_classes[c] = classes[ind]  # convert to class name

test['prediction'] = pred_all_classes

hist_pred = Counter(pred_all_classes)
print(hist_pred)

test.to_pickle("./files/predict_first_phase.pkl")  # save the fisrt phase prediction.

a = 1
