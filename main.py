import subprocess
import pandas as pd

# this is the main file:
# 1) we pre-process the training data
# 2) we build models (in 3 phases)
# 3) we use the models to predict the evaluation data classes and create the submission file.

print("preprocessing training data")
subprocess.call(["python", "preprocess_training_data.py"])

print("building models")
subprocess.call(["python", "build_models_run_all.py"])

print("predicting evaluation data and creating submission file")
subprocess.call(["python", "submit_run_all.py"])

# sanity: # we make sure that we re-created the submission file used in the ML challenge
print("compare submission file to the one previously saved")
eval1 = pd.read_csv("./files/submission.csv")
eval0 = pd.read_csv("./files/submission_0.731.csv")
print(eval0.shape)
print(eval1.shape)
assert(eval0.equals(eval1))

print("---------")
print("finished!")
print("---------")