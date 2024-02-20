# Supervised-Learning


Introduction

This project explores three classification algorithms - KNN, Logistic Regression, and Decision Tree - using various evaluation metrics. Libraries such as Numpy, Pandas, Matplotlib, Seaborn, and Sklearn are employed for numerical calculations, data manipulation, visualization, and building models. The dataset, comprising 700 training examples with two classes (459 in class 0 and 241 in class 1), is divided into 80% training and 20% testing data using Sklearn's train_test_split function.

KNN on Validation Data


For KNN, K-neighbour classifiers are utilized with k-fold cross-validation (k=5). The hyperparameter 'k' is chosen from the range of 1 to 31 (odd numbers only). After determining the best 'k' for each metric, the KNN model is trained with k=9, and evaluation metrics (accuracy, precision, recall, F1) are printed. The cross-validation scores for each 'k' are also plotted.

Logistic Regression on Validation Data


Logistic Regression is implemented with a range of hyperparameters (0.001, 0.01, 0.1, 1, 10, 100). After cross-validation, the best 'c' is selected (c=0.1), and the model is trained and evaluated. Cross-validation scores for each 'c' are plotted.

Decision Tree on Validation Data

Decision Tree is applied with hyperparameters ranging from 1 to 11. The best values for each metric are determined, and the model is trained with max depth=5. Training and evaluation metrics are printed, and cross-validation scores are plotted.

Evaluation


To obtain the best-performing version of each model, k-fold cross-validation is employed, and the best hyperparameter is selected based on accuracy, precision, recall, and F1-score. Confusion matrices are printed to evaluate the model's performance on test data, highlighting true positives, true negatives, false positives, and false negatives.

Results


While all algorithms perform well, KNN demonstrates superior performance based on the confusion matrix. True positive and true negative rates are higher, indicating better classification. Logistic Regression follows, and Decision Tree performs less effectively. Test data evaluation metrics further support the superiority of KNN in terms of accuracy, precision, recall, and F1-score.

Evaluation Metrics on Test Data:
KNN:

Test Accuracy: 0.9857
Test Recall: 0.9783
Test Precision: 0.9783
Test F1: 0.9783
Logistic Regression:

Test Accuracy: 0.9643
Test Recall: 0.9130
Test Precision: 0.9767
Test F1: 0.9438
Decision Tree:

Test Accuracy: 0.9500
Test Recall: 0.9565
Test Precision: 0.9263
Test F1: 0.8980


Conclusion

In conclusion, the evaluation criteria (accuracy, precision, recall, and F1-score) suggest that KNN outperforms Logistic Regression and Decision Tree on this dataset. KNN, being a non-parametric method, adapts well to non-linear data. Logistic Regression, a parametric method, excels with linearly separable data, while Decision Tree, an easy-to-understand method, performs well with non-linear data. Each method's effectiveness depends on dataset characteristics and classification objectives.

Reflecting on the project, future improvements could include more sophisticated hyperparameter tuning techniques, such as Bayesian optimization or random search. Overall, the project provided valuable insights into classification techniques and their evaluation using diverse metrics.
