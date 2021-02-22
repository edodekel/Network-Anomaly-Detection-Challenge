import pandas as pd
import numpy as np
import time
from collections import Counter
import pickle

t0 = time.time()
test = pd.read_pickle("./files/predict_second_phase.pkl")  # load previous phase predictions
t1 = time.time()
print(f"... total elapsed = {t1 - t0}")
print(test.shape)
# third_phase:
test_other2 = test[test.prediction == 'other2']  # prediction only the "other2" bucket
print(test_other2.shape)

with (open('./files/third_phase_models.pkl', "rb")) as openfile:  # load models
    while True:
        try:
            models = pickle.load(openfile)
        except EOFError:
            break

with (open('./files/third_phase_models_f_vec_optimized.pkl', "rb")) as openfile:  # load factor vec
    while True:
        try:
            f_vec_optimized = pickle.load(openfile)
        except EOFError:
            break

test_other2 = test_other2.drop(columns=['prediction'])
classes = models[0].classes_
num_classes = models[0].n_classes_

pred_all = np.zeros((test_other2.shape[0], num_classes))

c = -1
for model in models:
    c += 1
    print(f"model #{c}")
    y_pred = model.predict_proba(test_other2)  # sum the predicted probabilities of each model
    pred_all += y_pred

pred_all0 = pred_all.copy()

n = -1
for f in f_vec_optimized:
    n += 1
    pred_all[:, n] = pred_all[:, n] * f  # multiply each column by its corresponding factor from the factor vector

pred_all_ind = np.argmax(pred_all, axis=1)  # predict
pred_all_classes = np.array(['abbaaaaaaaaaaaa' for _ in range(len(pred_all_ind))])
c = -1
for ind in pred_all_ind:
    c += 1
    pred_all_classes[c] = classes[ind]  # convert to class name

test.loc[test.prediction == 'other2', 'prediction'] = pred_all_classes  # put back in original prediction

prediction = test.prediction
hist_pred = Counter(pred_all_classes)
print(hist_pred)

test.to_pickle("./files/predict_third_phase.pkl")  # save

a = 1
