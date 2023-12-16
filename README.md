
# Text Classification Model Analysis

## Overview
This project explores various machine learning models for text classification. We compare Logistic Regression, K-Nearest Neighbors, Decision Tree, and Random Forest classifiers across different feature set sizes to determine the most effective approach.

## Models Evaluated
- Logistic Regression
- K-Neighbors Classifier
- Decision Tree Classifier
- Random Forest Classifier

## Dataset
We used a dataset comprising sentences labeled by difficulty, sourced from [Training Data](https://raw.githubusercontent.com/DalipiDenis/assign/main/training_data.csv) and [Test Data](https://raw.githubusercontent.com/DalipiDenis/assign/main/unlabelled_test_data.csv).

## Methodology
Each model was trained and evaluated on feature sets of varying sizes (4000, 500, 1000, 3000 features). We assessed model performance using metrics like accuracy, precision, recall, and F1-score.

## Results

### Performance Table
| Model Type               | Max Features | Accuracy | Precision | Recall | F1-Score |
|--------------------------|--------------|----------|-----------|--------|----------|
| Logistic Regression      | 4000         | 45.31%   | 44.52%    | 45.31% | 44.46%   |
| Logistic Regression      | 500         |  38.64%     | 37.78%       | 38.64%    |37.83%       |
| Logistic Regression      | 1000         | 41.56%      | 41.01%       | 41.56%    | 40.95%      |
| Logistic Regression      | 3000         | 44.47%      | 43.66%       | 44.47%    | 43.69%      |
| K-Neighbors Classifier   | 4000         | 18.75%      | 32.13%       | 18.75%    |  9.40%     |
| K-Neighbors Classifier   | 500         | 24.06%      | 33.29%       | 24.06%    | 18.97%      |
| K-Neighbors Classifier   | 1000         | 24.37%      | 35.21%       | 24.37%    | 17.02%      |
| K-Neighbors Classifier   | 3000         | 19.27%      | 32.24%       | 19.27%    | 10.20%      |
| Decision Tree Classifier | 4000         | 31.97%      | 31.84%       | 31.97%    | 31.68%      |
| Decision Tree Classifier | 500         | 28.54%      | 28.71%       | 28.54%    | 28.55%      |
| Decision Tree Classifier | 1000         | 31.25%      | 31.33%       | 31.25%    | 31.13%      |
| Decision Tree Classifier | 3000         | 30.20%      | 30.27%       | 30.20%    | 30.10%      |
| RandomForestClassifier   | 4000         | 38.54%      | 38.34%       | 38.54%    | 37.40%      |
| RandomForestClassifier   | 500         | 37.18%      | 36.72%       | 37.18%    | 36.32%      |
| RandomForestClassifier   | 1000         | 37.18%      | 36.63%       | 37.18%    | 36.14%      |
| RandomForestClassifier   | 3000         | 40.52%      | 40.44%       | 40.52%    | 39.52%      |

### Best Model
The best-performing model was **Logistic Regression** with 4000 features, achieving an accuracy of **45.31%**, precision of **44.52%**, and an F1-score of **44.46%**.

## Analysis

### Confusion Matrices
Here are the confusion matrices for each model with 4000 features:

#### Logistic Regression (4000 Features)
![Logistic Regression Confusion Matrix](https://github.com/SunderAli416/french-text/blob/main/logistic_regression.png)

#### K-Neighbors Classifier (4000 Features)
![K-Neighbors Classifier Confusion Matrix](https://github.com/SunderAli416/french-text/blob/main/knn.png)

#### Decision Tree Classifier (4000 Features)
![Decision Tree Classifier Confusion Matrix](https://github.com/SunderAli416/french-text/blob/main/decision_tree.png)

#### RandomForestClassifier (4000 Features)
![RandomForestClassifier Confusion Matrix](https://github.com/SunderAli416/french-text/blob/main/random_forest.png)

### Erroneous Predictions
Examples of incorrect predictions:

1. **Example 1:**
   - Sentence: "..."
   - True Label: ...
   - Predicted Label: ...
   - Possible Reason: ...

2. **Example 2:**
   - Sentence: "..."
   - True Label: ...
   - Predicted Label: ...
   - Possible Reason: ...
  
3. **Example 3:**
   - Sentence: "..."
   - True Label: ...
   - Predicted Label: ...
   - Possible Reason: ...  

