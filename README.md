# Credit Risk Analysis – A Practical Application of Sampling Methods, Logistic Regression and Ensemble Learners 

## Overview

Machine learning is a branch of artificial intelligence wherein algorithms are used to imitate the way that humans learn to build a model that describes a data set and, ideally, gradually improves the accuracy of predictions that can be made about alternate, analogous, data sets.  Machine learning can be broadly divided into the categories of “Unsupervised Learning” – where there is no specific outcome/output associated with the data set and the objective is to group the data into similar “clusters” – and “Supervised Learning” – where there is a specific outcome/output associated with the data set and the objective is to make predictions about the outcome/output associated with future data sets.  Within the category of “Supervised Learning”, the algorithms used to generate models are divided into “Linear Regression” – used when the outcome/output measured is a continuous variable – and “Logistic Regression” – used when the outcome/output measure is a discrete or binary variable – and in both cases, there are various measures of “fit” that can be used to measure/describe the applicability of the model developed to the data set in question.  The purpose of this analysis was to apply a number of different “Logistic Regression” models to lending data from “The Lending Club” in the interests of predicting credit card (default) risk and then, using the accepted measures “balanced accuracy”, “precision” and “recall”, determine which model(s), if any, should be deployed to assess/predict credit (default) risk for future lending scenarios.  

## Results

To begin, the Lending Club data was imported into a Jupyter Notebook in a python machine learning environment and subsets of the data with formatting or data types which would have interfered with/impeded the analysis process were transformed.  To initiate the actual analysis, and “supervised learning” process, the target variable, “loan status”, was isolated and the sklearn “train_test_split” package was used to split the data into training and testing subsets.
Once the data had been split in training and test subsets, the data was resampled with the imblearn RandomOverSampler, modeled with the sklearn logistic regression model and the confusion matrix, balanced accuracy score and imbalanced classification report associated with the model were produced (see Figure 1).

### Figure 1: NAIVE RANDOM OVERSAMPLING SUMMARY
![]( https://github.com/Scruffy-Bearie/Credit_Risk_Analysis/blob/main/IMAGES/Image1.png)

For the sake of comparing the results acquired with different over-sampling methods, the data set was then again oversampled using imblearn SMOTE, modeled with the sklearn logistic regression model and the confusion matrix, balanced accuracy score and imbalanced classification report associated with the model were produced (see Figure 2).

### Figure 2: SMOTE OVERSAMPLING SUMMARY
![]( https://github.com/Scruffy-Bearie/Credit_Risk_Analysis/blob/main/IMAGES/Image2.png)

So that results associated with oversampling of the data set could be compared to those associated with undersampling, the data set was undersampled using imblearn ClusterCentroids, modeled with the sklearn logistic regression model and the confusion matrix, balanced accuracy score and imbalanced classification report associated with the model were produced (see Figure 3).

### Figure 3: UNDERSAMPLING SUMMARY
![]( https://github.com/Scruffy-Bearie/Credit_Risk_Analysis/blob/main/IMAGES/Image3.png)

Finally, the data was “combination” sampled using imblearn SMOTEEN, modeled with the sklearn logistic regression model and the confusion matrix, balanced accuracy score and imbalanced classification report associated with the model were produced (see Figure 4).

### Figure 4: COMBINATION (SMOTEEN) SAMPLING SUMMARY
![]( https://github.com/Scruffy-Bearie/Credit_Risk_Analysis/blob/main/IMAGES/Image4.png)

To extend the analysis even further, the data set was modeled with the imblearn ensemble classifiers “Balanced Random Forest” and “AdaBoost” and the confusion matrix, balanced accuracy score and imbalanced classification report associated with the models were produced (see Figures 5 and 6).

### Figure 5: BALANCED RANDOM FOREST CLASSIFIER SUMMARY
![]( https://github.com/Scruffy-Bearie/Credit_Risk_Analysis/blob/main/IMAGES/Image5.png)

### Figure 6: EASY ENSEMBLE AdaBOOST CLASSIFIER SUMMARY
![]( https://github.com/Scruffy-Bearie/Credit_Risk_Analysis/blob/main/IMAGES/Image6.png)

In the interests of being able to easily compare all of the models employed/tested, the results associated with models were extracted for presentation in bulleted list format:

#### •	NAÏVE RANDOM OVERSAMPLING:
- Balanced accuracy: 0.67
- Precision: 0.01 (1.00)
- Recall: 0.71 (0.64)

#### •	SMOTE OVERSAMPLING:
-	Balanced accuracy: 0.64
-	Precision: 0.01 (1.00)
-	Recall: 0.61 (0.67)

#### •	UNDERSAMPLING:
-	Balanced accuracy: 0.54
-	Precision: 0.01 (1.00)
-	Recall: 0.68 (0.40)

#### •	COMBINATION (SMOTEEN) SAMPLING:
-	Balanced accuracy: 0.65
-	Precision: 0.01 (1.00)
-	Recall: 0.74 (0.57)

#### •	BALANCED RANDOM FOREST CLASSFIER:
-	Balanced accuracy: 0.99
-	Precision: 0.9 (1.0)
-	Recall: 1.0 (1.0)

#### •	AdABOOST CLASSIFIER:
-	Balanced accuracy: 1.0
-	Precision: 1.0 (1.0)
-	Recall: 1.0 (1.0)


## Summary

A quick review of the results for all models, Figures 1 through 6 and bulleted list included in “Results” section, evidences that there was no dramatic difference in the “balanced accuracy” for the models resulting from the different sampling methods (scores from 0.54 for “Undersampling” to 0.67 for “Naïve Random Oversampling”), that “precision” for the different models was identical (0.01 (1.0)) and that “recall” ranged from 0.61 for “SMOTE Oversampling” to 0.74 for “SMOTEEN” combination sampling.  In contrast, the models produced using the ensemble classifiers resulted in “balanced accuracy” scores of 0.99 for the Balanced Random Forest Classifier and 1.0 for the AdaBoost Classifier, “precision” scores of 0.9 and 1.0 respectively and a “recall” score of 1.0 for both models.  Before using these metrics to compare the models and decide which of the models, if any, should be adopted by LendingClub, it seemed important to compare what the metrics actually measure to the business objectives and risks for LendingClub.

Given the nature of the data set, and the fact that “Low_Risk” loans far outnumber “High_Risk” loans, it was decided that the “Balanced Accuracy” score was not necessarily the best indicator for comparing the relative usefulness of the models developed.  It was also decided that wherein identifying “Low_Risk” loans as “High_Risk” (false positives) could have a negative impact on revenues (as fewer loans would be granted), that it was likely far more important that all “High_Risk” loans are identified correctly.  As such, when evaluating the relative usefulness of the models developed as determined through comparison of the metrics associated with those models, it was decided that “Recall” – the ability of the model to find all positive samples – was of greater importance than “Accuracy” and “Precision”.

All that said, of the models associated with different sampling methods, SMOTEEN (or combination sampling) produced a model with the highest “Recall” score (0.74) but it was (obviously) the ensemble learners, specifically the “AdaBoost Classifier”, that produced the best/most reliable models overall.  Wherein it would be tempting to suggest that LendingClub adopt the “AdaBoost Classifier” model for client/loan application screening “immediately”, the AdaBoost algorithm has been demonstrated to “overfit” some data sets – primarily those that are “noisy” or contain a high proportion of outliers.  As such, it is recommended or suggested, that the data set in question be subjected to greater scrutiny before the “AdaBoost” model generated using that data set be adopted for client/loan application screening.

