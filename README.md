# LAweek9Q3

Wine Quality Classification
# Dataset
Files used: winequality-red.csv, winequality-white.csv, winequality.names  

Dataset characteristics: 4898 instances, 11 input features, 1 output variable, no missing values  
The quality score is converted into binary classification with Good wine being a quality score > 6, and Bad wine being a quality score <= 6  

# Workflow
1. Load white and red wine datasets
2. Combine datasets
3. Create binary target variable
4. Split the data into: Training(70%), Validation set(15%), Test set(15%)
5. Scale features with standardscaler
6. Implement the 3 algorithms
7. Evaluate the data

#Required pyhton packages  
Python 3.xx
pandas  
scikit-learn  
ucimlrepo  
installable with: pip install pandas scikit-learn ucimlrepo  

# Running the Code  

1. download or clone this repo
2. navigate to the directory
3. enter the following command into your terminal or powershell: python week9LAQ3.py

# Understanding the output  
program shows 2 evaluation outputs, one displaying the validation set and the other the test set  
Validation set is used to compare models and tune parameters better for better results  
Test set shows the final performance results on unseen data  

# analysis  
Decision Tree: Best performing due to overall balance between results and had the Highest F1 score  
KNN: Similar accuracy to decision tree with the Lower recall  
Naive Bayes: Weak precision and predicts too many wines as good  

