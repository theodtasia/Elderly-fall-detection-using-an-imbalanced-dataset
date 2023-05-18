# Elderly-fall-detection-using-an-imbalanced-dataset


## Description
The aim of this work is to use an imbalanced time-series dataset for anomaly detection of elderly people, using machine learning algorithms such as Decision Tree, SVM, Random Forest, KNN, Naive Bias and ensembles models to detect the falls using the data from 4 sensors. 
Techniques are applied to deal with class imbalance and we will build a cost matrix to deepen the research in the domain of cost-sensitive learning.
Finally, the machine learning models will be compared in order to extract the most valuable conclusions and important results for detecting elderly fall events

##Dataset
The dataset can be found here: https://www.kaggle.com/datasets/jorekai/anomaly-detection-falling-people-events

Regarding the labels, they represent a fall or a normal activity, where zero-(0) means normal and one-(1) means anomalous fall event.
Below there is a bar chart presenting the issue of class imbalanced in the dataset.
![s3](https://github.com/theodtasia/Elderly-fall-detection-using-an-imbalanced-dataset/assets/36897965/0f5cd506-4e41-4c01-afac-d7afc7c5ac21)

## Results

# Class Imbalance

First in the ranking is SMOTE technique with Random Forest as it performed better in all metrics with 89% in F1-score, 74% in G-mean and 55% in imbalanced accuracy. 
Slightly behind is SMOTE with Logistic Regression with 83% in F1-score, 74% in G-mean and 56% in imbalanced accuracy.
Easy Enseble performed the greatest F1-score 90%, but slightly lower than SMOTE in the other two metrics with 67% in G-mean and 47% in imbalanced accuracy in all models. 
Near Miss 1 technique seemed to be not suitable to be used as an imbalanced technique, as it achieved on average of the four tested models 29% in imbalanced accuracy 53,4% in G-mean and 52,7% in F1-score.

![cq](https://github.com/theodtasia/Elderly-fall-detection-using-an-imbalanced-dataset/assets/36897965/a9937e51-4fd5-4935-a29d-064bc1d33b73)

# Cost sensitive learning

The results without sampling but using a simple random forest model gave a cost of 7234 with FN to be 1250 cases.
Using the under-sampling method, the loss was reduced at 6775 and the FN to 405. There reduction with under-sampling was significant big, it was a 308% reduction!
The combination of under-sampling and oversampling was very close to the under-sampling method. 
The weight method had the second best result, as the method gave an important minimum reduction of the cost.

![c3](https://github.com/theodtasia/Elderly-fall-detection-using-an-imbalanced-dataset/assets/36897965/6d54143b-3335-4be9-9b7d-02aee8cb1f41)

##Conclusions
Built a cost matrix, [[0, 1], [5, 0]], to deepen the research in the domain of cost-sensitive learning.
The results showed that Smote technique gave the best results in balanced accuracy by running an Ensemble Classifier and Random Forest classifier. 
The best method in  cost sensitive learning was the under-sampling technique which minimized the loss and also the False Negatives up to 308% compared to without sampling method.
