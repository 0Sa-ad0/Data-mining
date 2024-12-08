import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import kagglehub

# Download latest version
dataset_path = kagglehub.dataset_download("lara311/diabetes-dataset-using-many-medical-metrics")

print("Path to dataset files:", dataset_path)

# Assuming the main dataset is in a CSV file, update the filename to the correct one
dataset_file = f"{dataset_path}/diabetes (1).csv"  # Updated filename

# Load the dataset into a DataFrame
dataset = pd.read_csv(dataset_file)

# Inspect the dataset structure and summary
print(dataset.info())
print(dataset.describe())

# Handle missing values (if any)
dataset.fillna(0, inplace=True)

# Ensure the target variable is categorical
dataset['Outcome'] = dataset['Outcome'].astype('category')

# Split the data into training and testing sets
train_data, test_data = train_test_split(dataset, test_size=0.3, random_state=123)

# Function to calculate the mean and standard deviation for each class
def calc_params(data):
    params = {}
    for class_label in data['Outcome'].cat.categories:
        class_data = data[data['Outcome'] == class_label]
        params[class_label] = {col: {'mean': class_data[col].mean(), 'sd': class_data[col].std()} 
                               for col in data.columns if col != 'Outcome'}
    return params

# Calculate the parameters for the Naive Bayes model
params = calc_params(train_data)

# Function to calculate the probability density function
def dnorm_custom(x, mean, sd):
    return (1 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / sd) ** 2)

# Function to make predictions
def predict_nb(test_data, params, train_data):
    predictions = []
    for _, row in test_data.iterrows():
        probs = {}
        for class_label, class_params in params.items():
            prob = 1
            for col, stats in class_params.items():
                prob *= dnorm_custom(row[col], stats['mean'], stats['sd'])
            prob *= len(train_data[train_data['Outcome'] == class_label]) / len(train_data)
            probs[class_label] = prob
        predictions.append(max(probs, key=probs.get))
    return pd.Categorical(predictions, categories=params.keys())

# Make predictions on the test set
test_data_no_outcome = test_data.drop(columns=['Outcome'])
predictions = predict_nb(test_data_no_outcome, params, train_data)

# Evaluate the model
conf_matrix = confusion_matrix(test_data['Outcome'], predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate accuracy
accuracy = accuracy_score(test_data['Outcome'], predictions)
print(f"Accuracy: {accuracy:.2f}")
