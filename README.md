
## Summary ##

The challenge is a classification problem with very unbalanced amounts of data of each class. Later it was revealed that some of the evaluation data is "true" anomalies, i.e. not persent in the training set, but we didn't have time to change the process accordingly.

To tackle the different amount of data of each class, we build models in 3 phases, where in the first two phases we bucket together the smaller classes into one class to make the classes not too different in size.

In each phase we use OneVsRestClassifier to train several models at once where the classifier used by OneVsRestClassifier is a random forest classifier with weighting of the samples inversely proportional to the amount of samples of that class.

We optimize weights to multiply the predicted probability for the smaller classes (and for the buckets in the 2 first phases) in order to improve the over-all balanced accuracy

A caveat: we run the complete process with several seeds. we are not completely satisfied with variablity of the results. We didn't have time to investigate it further and hope that the leader-board results would still translate to the rest of the test data.

Remark: Disregarding the "normal" class could have increased the balanced accuracy significantly. But it would have defeated the purpose of the exercise (to detect "anomalies"). In a real-life scenario false alarms have significant costs (alert fatigue and bigger analysis teams). Same could perhaps be said that "optimizing the weights" in order to improve the over-all balanced accuracy, but to a lesser extent.
A better accuracy measure would have been to give the true positive rate for "normal" a much higher weight in the balanced accuracy (or multiply the "normal" true positive rate by the balanced accuracy of the rest, etc...). 

## Preprocessing & Feature Selection / Engineering ##

We tested the amount of samples of each class in the train set. We assume that the test set has similar distribution of classes (at least for the smaller classes).

We converted columns of type categorical and columns with less than 11 values to one-hot columns.

We reduce the total memory size by convertin to int32 and float32.

## Training methodology ##
We classify in three phases:
* phase 1:

We classify normal, class10 and class18 (the 3 largest classes) and we bucket all the rest as "other". In this way the smallest ("other") class is not too small.
for each of 4 cross-validation folds we randomly select 100,000 samples of the 3 largest classes and 30,000 of "other" (we don't use all of the data here).
We use OneVsRestClassifier to build a model, where we use a Random Forest classifier as the inner classifier. We weight each sample inversely proportional to the amount of its original class.
We test on random samples of the 3 classes (normal,class10 and class18) and the rest of the samples of the "other" class. Prediction (predict_probability) gives an array of size number of test samples X 4.
We multiply the last column ("other") by a factor so as to reduce the number of "other" errors (this increases in turn the number of "normal" errors). The factor is selected so as to have zero errors in class "other". 
For each row the prediction is the index of the maximum of that row.

* phase 2: 

We classify all the classes with more than 950 samples except "normal", "class10" and "class18" (the 3 largest classes which were classified in the previous phase). We also add here"class07" and "class14" which were proved to be prediced without any errors during an investagation of the data. We bucket all the rest as "other2". 

We now have enough samples of each class to divide into 8 folds. We use MultilabelStratifiedKFold algorithm (see reference below) which divide the train samples into folds in an optimized fashion.

For each fold we use the samples of that fold as test data and the rest as train data to build models. We use OneVsRestClassifier to build the models, where we use a Random Forest classifier as the inner classifier. We weight each sample inversely proportional to the amount of its original class.

Prediction (predict_probability) gives an array of size number of test samples X number of classes in this phase + 1 (for "other2").
we multiply the last column ("other2") by a factor so as to reduce the number of "other2" errors. The factor is selected so as to maximize the balanced accuracy (with higher weight for class "other2").
For each row the prediction is the index of the maximum of that row.

* phase 3:

We classify all the classes not classified in the previous phases (158 samples). We run 100 folds. We select ~2/3 of the samples of each class and test on the the rest. Again, we use OneVsRestClassifier to build the models, where we use a Random Forest classifier as the inner classifier. We weight each sample inversely proportional to the amount of its original class. We use the test data of the first 50 folds to optimize a factor vector used to multiply each column of the prediction matrix. The target function for optimization is the balanced accuracy and we optimize by Powell's method (without gradients). We use the other 50 folds to validated that the balanced accuracy increased.

## Notable aspects ##
the larger classes are "easier" to classify. therefore:

* (1a) When predicting, we increase the weight of the smaller classes in order to decrease their error while increasing the error of the larger classes => better balanced accuracy
* (1b) When building models, using weights for each sample according to its class (higher weight for the smaller classes) reduces the error for the smaller classes => better balanced accuracy
* (2) we optimize the weights for class selection in order to reduce the final balanced accuracy (see 1a).

## References ##
OneVsRestClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html#sklearn.multiclass.OneVsRestClassifier
Sample weighting in OneVsRestClassifier: https://stackoverflow.com/questions/49534490/use-sample-weight-in-multi-label-classification
MultilabelStratifiedKFold:  https://github.com/trent-b/iterative-stratification
Powell's Method: https://en.wikipedia.org/wiki/Powell%27s_method
## Inline code comments ##
code may be found at: https://github.com/edodekel/Network-Anomaly-Detection-Challenge

Please put "training.csv" in "files" folder before running the script.
Run "main.py"
