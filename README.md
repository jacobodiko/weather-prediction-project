# Weather Prediction Model Using Decision Tree Classifier

## Overview

This project aims to build a **weather prediction model** that predicts whether it will rain on a given day based on historical weather data. The model uses a **Decision Tree Classifier** from the **scikit-learn** library. This is a beginner-friendly project that covers essential machine learning concepts, including data preparation, feature engineering, model training, prediction, and evaluation.

The project walks through each step of the machine learning workflow, from preparing and manipulating data to building and evaluating a machine learning model. The final result includes performance metrics such as accuracy, a confusion matrix, and classification report, which provide insights into the model's effectiveness.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Dependencies](#dependencies)
3. [Dataset](#dataset)
4. [Data Preparation](#data-preparation)
5. [Feature Engineering](#feature-engineering)
6. [Building the Model](#building-the-model)
7. [Model Evaluation](#model-evaluation)
8. [Usage](#usage)
9. [License](#license)

---

## Project Structure

The main files and folders in this project are:

```
weather-prediction-project/
│
├── weather_prediction_model.ipynb  # Jupyter notebook containing the code and explanations
├── README.md                       # Detailed project documentation
└── data/
    └── sample_weather_data.csv      # Sample dataset used in this project
```

---

## Dependencies

The project relies on the following Python libraries:

- **pandas**: Data manipulation and analysis.
- **numpy**: Numerical operations.
- **scikit-learn**: Machine learning algorithms and utilities.
- **seaborn**: Data visualization.
- **matplotlib**: Plotting graphs and visualizations.

You can install the dependencies using `pip`:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

---

## Dataset

We are working with a small, simulated weather dataset that includes the following features:

- **day_of_week**: The day of the week (e.g., Monday, Tuesday).
- **temperature**: The temperature in degrees Celsius.
- **humidity**: The percentage of humidity in the air.
- **wind_speed**: The wind speed in km/h.
- **pressure**: The atmospheric pressure in hPa.
- **rain**: The target variable that indicates whether it rained (1) or not (0).

Here is a sample of the dataset:

| day_of_week | temperature | humidity | wind_speed | pressure | rain |
|-------------|-------------|----------|------------|----------|------|
| Monday      | 20          | 65       | 10         | 1012     | 1    |
| Tuesday     | 25          | 70       | 5          | 1018     | 0    |
| Wednesday   | 18          | 80       | 12         | 1009     | 1    |
| Thursday    | 22          | 50       | 9          | 1010     | 0    |
| Friday      | 28          | 55       | 15         | 1013     | 0    |
| Saturday    | 24          | 60       | 6          | 1020     | 0    |
| Sunday      | 19          | 85       | 8          | 1016     | 1    |

The dataset is small for illustration purposes, but in a real-world project, you would work with larger and more diverse datasets.

---

## Data Preparation

### 1. Import Libraries

We start by importing all the necessary libraries:

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

### 2. Load the Data

Load the dataset into a pandas DataFrame:

```python
data = {
    'day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    'temperature': [20, 25, 18, 22, 28, 24, 19],
    'humidity': [65, 70, 80, 50, 55, 60, 85],
    'wind_speed': [10, 5, 12, 9, 15, 6, 8],
    'pressure': [1012, 1018, 1009, 1010, 1013, 1020, 1016],
    'rain': [1, 0, 1, 0, 0, 0, 1]
}
df = pd.DataFrame(data)
```

### 3. Data Exploration

Before building the model, we explore the data to understand the distribution of values:

```python
df.info()          # Get basic information about the dataset
df.describe()      # Get statistical details
sns.pairplot(df, hue='rain')  # Visualize the relationships between features
plt.show()
```

---

## Feature Engineering

To make the dataset compatible with machine learning algorithms, we convert the categorical variable `day_of_week` into numerical format using **one-hot encoding**:

```python
df = pd.get_dummies(df, columns=['day_of_week'])
```

Now the dataset contains binary (0 or 1) columns for each day of the week.

---

## Building the Model

### 1. Splitting the Data

We split the dataset into features (`X`) and target (`y`), then further split the data into training and testing sets:

```python
X = df.drop('rain', axis=1)
y = df['rain']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 2. Training the Decision Tree Classifier

We create and train the **Decision Tree Classifier** model:

```python
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
```

---

## Model Evaluation

### 1. Making Predictions

After training the model, we make predictions on the test data:

```python
y_pred = model.predict(X_test)
```

### 2. Evaluating the Model

We evaluate the model’s performance using the following metrics:

- **Accuracy**: The percentage of correct predictions.
- **Confusion Matrix**: Shows the breakdown of correct and incorrect predictions for each class.
- **Classification Report**: Provides precision, recall, and F1-score for each class.

```python
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)
```

---

## Expected Results and Interpretations

- **Accuracy**: This shows how many of the predictions were correct. For example, an accuracy of `0.75` means that 75% of the predictions were correct.
  
- **Confusion Matrix**: The confusion matrix helps us see exactly how the model performed in terms of True Positives, True Negatives, False Positives, and False Negatives.

- **Classification Report**: This report includes metrics like **precision**, **recall**, and **F1-score**, which help us understand the model's performance in predicting both "rain" and "no rain" days.

---

## Usage

To run this project:

1. Clone this repository:
   ```bash
   git clone https://github.com/jacobodiko/weather-prediction-project.git
   cd weather-prediction-project
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook `weather_prediction_model.ipynb` to see the model in action.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Conclusion

This project demonstrates the basic workflow for building a machine learning model to predict whether it will rain, based on weather data. The steps include data preprocessing, feature engineering, model training, and evaluation using common metrics such as accuracy, confusion matrix, and classification report.