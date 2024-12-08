# DATA MINING 

# Diabetes Prediction Using Medical Metrics

This project aims to predict the likelihood of diabetes based on various medical metrics using machine learning techniques. The dataset used is from a public source and contains data on medical attributes such as glucose levels, BMI, and age, with a target variable indicating whether the patient has diabetes.

## Dataset Information

- **Total Entries**: 768
- **Total Features**: 9 columns including medical attributes like `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`, and the target variable `Outcome`.
- **Target Variable (`Outcome`)**: Indicates whether the patient has diabetes (`1` for positive, `0` for negative).

## Data Preprocessing

- **Missing Values**: Missing values were handled by filling them with zeros to avoid errors during model training.
- **Feature Inspection**: The dataset contains a mix of integer and float features, with some features like `Insulin` and `BloodPressure` containing zero values, which could indicate missing data.
- **Data Split**: The dataset was split into training and testing sets with a 70-30 ratio for model evaluation.

## Model Used

- **Naive Bayes Classifier**: A probabilistic model was used to classify the data based on the features. The Naive Bayes algorithm is simple yet effective for classification problems, assuming feature independence.

## Evaluation Metrics

- **Accuracy**: The model achieved an accuracy of **78%** on the test set.
- **Confusion Matrix**:

- **True Negatives**: 124
- **False Positives**: 19
- **False Negatives**: 32
- **True Positives**: 56

## Conclusion

- The Naive Bayes model performed reasonably well with a **78% accuracy**.
- There were some misclassifications (19 false positives and 32 false negatives), but the model still successfully predicted diabetes in a significant portion of cases.
- Future improvements could include feature engineering, exploring other machine learning algorithms, and tuning the model for better performance.

## Technologies Used

- **Python**: The project was developed using Python, utilizing libraries like Pandas, NumPy, and Scikit-Learn.
- **Scikit-Learn**: Used for implementing the Naive Bayes classifier and evaluating the model.

## How to Run
 Clone this repository:
 ```bash
 git clone https://github.com/yourusername/Data-mining.git
