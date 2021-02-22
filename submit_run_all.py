import time
import pandas as pd
import subprocess

print("Starting predict_first_phase")
subprocess.call(["python", "predict_first_phase_with_thresholds.py"])
print("Starting predict_second_phase")
subprocess.call(["python", "predict_second_phase.py"])
print("Starting predict_third_phase")
subprocess.call(["python", "predict_third_phase.py"])

t0 = time.time()
test = pd.read_pickle("./files/predict_third_phase.pkl")  # read final prediction file
t1 = time.time()
print(f"... total elapsed = {t1 - t0}")
print(test.shape)

X = test[["prediction"]]
X.to_csv("./files/submission.csv")  # save to csv

print("submission file is ready")
