import pandas as pd
import numpy as np
import time
import pickle
from collections import Counter

t0 = time.time()
test = pd.read_pickle("./files/predict_first_phase.pkl") # load the first phase prediction
t1 = time.time()
print(f"... total elapsed = {t1 - t0}")
print(test.shape)

# second_phase:
test_other = test[test.prediction == 'other'] # we now work only on the "other" bucket
print(test_other.shape)

with (open('./files/second_phase_models.pkl', "rb")) as openfile: # load models
    while True:
        try:
            models = pickle.load(openfile)
        except EOFError:
            break

with (open('./files/second_phase_models_factor_max.pkl', "rb")) as openfile: # load the factor of the "other2" class
    while True:
        try:
            factor_max = pickle.load(openfile)
        except EOFError:
            break

test_other = test_other.drop(columns=['prediction'])
classes = models[0].classes_
num_classes = models[0].n_classes_

pred_all = np.zeros((test_other.shape[0], num_classes))

c = -1
for model in models:
    c += 1
    print(f"model #{c}")
    y_pred = model.predict_proba(test_other) # sum the predicted probability of each model
    pred_all += y_pred

pred_all = np.array(pred_all)
L = pred_all.shape[1]
factor = factor_max * 1.25  # we add 25 percent to be "on the safe side"
#factor = 3
pred_all[:, L - 1] = np.array(pred_all[:, L - 1] * factor) # we multiply the last class ("other2") by the factor


pred_all_ind = np.argmax(pred_all, axis=1) # predict
pred_all_classes = np.array(['abbaaaaaaaaaaaa' for _ in range(len(pred_all_ind))])
c = -1
for ind in pred_all_ind:
    c += 1
    pred_all_classes[c] = classes[ind]  # convert to class name

hist_pred = Counter(pred_all_classes)
print(hist_pred)

test.loc[test.prediction == 'other', 'prediction'] = pred_all_classes # put back the predictions in the original prediction file
test.to_pickle("./files/predict_second_phase.pkl") # save prediction

a = 1
