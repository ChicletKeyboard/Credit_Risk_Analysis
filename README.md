# Credit_Risk_Analysis
Using supervised machine learning algorithms to test for accuracy of analyzing credit risk.

## Analysis Overview

In this project, we use Python to build and evaluate several machine learning models to predict credit risk.
We adopted the following procedure:

* Oversample the data using the RandomOverSampler and SMOTE algorithms.
* Undersample the data using the ClusterCentroids algorithm.
* Use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm.
* Compare two machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier.
* We will evaluate the performance of these models and make a recommendation on whether they should be used to predict credit risk.

## Results

### Random OverSampler Model
![Random Oversampler Risk](https://github.com/ChicletKeyboard/Credit_Risk_Analysis/blob/a820f17fe5df8a1f8519ad4e867ffef944eed2e1/Resources/Random%20OverSampler%20Model_1.jpg)
![Random Oversampler Table](https://github.com/ChicletKeyboard/Credit_Risk_Analysis/blob/a820f17fe5df8a1f8519ad4e867ffef944eed2e1/Resources/Random%20OverSampler%20Model.jpg)

* The balanced accuracy score is 65%.
* The high_risk precision is about 1% only with 62% sensitivity which makes a F1 of 2% only.
* Due to the high number of the low_risk population, its precision is almost 100% with a sensitivity of 68%.

### SMOTE model
![SMOTE Risk](https://github.com/ChicletKeyboard/Credit_Risk_Analysis/blob/a820f17fe5df8a1f8519ad4e867ffef944eed2e1/Resources/SMOTE_Model_1.jpg)
![SMOTE Table](https://github.com/ChicletKeyboard/Credit_Risk_Analysis/blob/a820f17fe5df8a1f8519ad4e867ffef944eed2e1/Resources/SMOTE_Model.jpg)

The results are pretty similar to the previous model:
* Balanced accuracy score is 64%.
* High_risk precision is about 1% only with 63% sensitivity which makes a F1 of 2% only.
* Due to the high number of the low_risk population, its precision is almost 100% with a sensitivity of 66%.

### ClusterCentroids Model

![ClusterCentroids Risk](https://github.com/ChicletKeyboard/Credit_Risk_Analysis/blob/a820f17fe5df8a1f8519ad4e867ffef944eed2e1/Resources/ClusterCentroids_Model_1.jpg)
![ClusterCentroids Table](https://github.com/ChicletKeyboard/Credit_Risk_Analysis/blob/a820f17fe5df8a1f8519ad4e867ffef944eed2e1/Resources/ClusterCentroids_Model.jpg)

* Here the balanced accuracy score is down to about 52%.
* The high_risk precision is still 1% only with 63% sensitivity which makes a F1 of 1%.
* Due to the high number of false positives, the low_risk sensitivity is only 40%.

### Combination (Over and Under) Sampling

![SMOTEENN Model Risk](https://github.com/ChicletKeyboard/Credit_Risk_Analysis/blob/a820f17fe5df8a1f8519ad4e867ffef944eed2e1/Resources/SMOTEENN_Model_1.jpg)
![SMOTEENN Model Table](https://github.com/ChicletKeyboard/Credit_Risk_Analysis/blob/a820f17fe5df8a1f8519ad4e867ffef944eed2e1/Resources/SMOTEENN_Model.jpg)

* Balanced accuracy score is about 62%.
* High_risk precision is still 1% only with 68% sensitivity which makes a F1 of only 2%.
* Due to the high number of false positives, the low_risk sensitivity is 57%.

### Balanced Random Forest Classifier Model

![Forest Risk](https://github.com/ChicletKeyboard/Credit_Risk_Analysis/blob/a820f17fe5df8a1f8519ad4e867ffef944eed2e1/Resources/BalancedRandomForestClassifier_Model_1.jpg)
![Forest Table](https://github.com/ChicletKeyboard/Credit_Risk_Analysis/blob/a820f17fe5df8a1f8519ad4e867ffef944eed2e1/Resources/BalancedRandomForestClassifier_Model.jpg)

* Balanced accuracy score improved to about 79%.
* High_risk precision is still low at 4% only with 67% sensitivity which makes a F1 of only 7%.
* Due to a lower number of false positives, the low_risk sensitivity is now 91% with 100% presicion.

### Easy Ensemble Classifier Model

![Easy Risk](https://github.com/ChicletKeyboard/Credit_Risk_Analysis/blob/a820f17fe5df8a1f8519ad4e867ffef944eed2e1/Resources/EasyEnsembleClassifier_Model_1.jpg)
![Easy Table](https://github.com/ChicletKeyboard/Credit_Risk_Analysis/blob/a820f17fe5df8a1f8519ad4e867ffef944eed2e1/Resources/EasyEnsembleClassifier_Model.jpg)

* Now, the balanced accuracy score is high to about 93%.
* The high_risk precision is still low at 7% only with 91% sensitivity which makes a F1 of only 14%.
* Due to a lower number of false positives, the low_risk sensitivity is now 94% with 100% presicion.

## Summary

All the models used to perform the credit risk analysis show weak precision in determining if a credit risk is high. The Ensemble models brought a lot more improvment specially on the sensitivity of the high risk credits. The EasyEnsembleClassifier model shows a recall of 92% so it detects almost all high risk credit. On another hand, with a low precision, a lot of low risk credits are still falsely detected as high risk which would penalize the bank's credit strategy and infer on its revenue by missing those business opportunities. For those reasons I would not recommend the bank to use any of these models to predict credit risk.
