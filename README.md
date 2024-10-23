# Mental Health Prediction Using Machine Learning

## Project Overview

This project was developed as part of a take-home task for a Data Scientist position at AXA Health. The task aims to predict mental health issues in individuals using machine learning models. The dataset includes demographic and behavioral attributes, and the goal is to use this information to create an effective prediction model for identifying individuals at risk of developing mental health issues.

## Key Objectives

1. **Data Analysis & Insights**: Perform exploratory data analysis (EDA) to uncover trends, correlations, and anomalies.
2. **Feature Engineering & Preprocessing**: Apply necessary transformations and scaling to both categorical and numerical features.
3. **Model Development & Selection**: Experiment with different machine learning models and select the best performing model based on key metrics such as accuracy, precision, recall, and F1-score.
4. **Model Evaluation**: Evaluate the model on a held-out test set and assess the model's limitations and biases.
5. **Final Model**: Develop a final model with the best-performing hyperparameters and assess its performance.

## Dataset

The dataset contains the following features:
- **Numerical features**: Age, Number of Children, Income (log-transformed).
- **Categorical features**: Marital Status, Education Level, Smoking Status, Physical Activity Level, Employment Status, Alcohol Consumption, Dietary Habits, Sleep Patterns, History of Substance Abuse, Family History of Depression, Chronic Medical Conditions.
- **Target variable**: History of Mental Illness (Yes/No).

## Project Structure
```bash
├── data
│   ├── raw
│   └── processed
├── notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_Model_Training.ipynb
├── src
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── utils.py
├── test
│   ├── test_model.py
├── README.md
├── requirements.txt
└── environment.yml
```




### Files and Directories
- **data/**: Contains the dataset used for training and testing the models.
- **src/**: Contains the Python scripts used for feature engineering, model development, and evaluation.
- **test/**: Includes notebooks with initial experiments, hyperparameter tuning, and model selection.
- **requirements.txt**: List of dependencies needed to run the project using pip.
- **environment.yml**: A Conda environment file for setting up the environment.
- **results/**: Contains plots, model outputs, and final evaluation metrics.

## Libraries and Packages

Here’s a list of the primary libraries and packages used in this project:
- **numpy**: 1.26.4
- **pandas**: 2.2.2
- **scikit-learn**: 1.5.1
- **imbalanced-learn**: 0.12.3
- **lightgbm**: 4.5.0
- **matplotlib**: 3.9.2
- **seaborn**: 0.13.2
- **optuna**: 4.0.0
- **joblib**: 1.4.2
- **tqdm**: 4.66.5
- **lime**: 0.2.0.1
- **shap**: 0.46.0
- **missingno**: 0.5.2

## Model Development Process

1. **Exploratory Data Analysis (EDA)**:
   - Trends and insights were derived using summary statistics and visualizations.
   - Correlation analysis and Chi-Square tests were performed to assess feature importance.

2. **Data Preprocessing**:
   - Numerical features were scaled using `StandardScaler`, and categorical features were encoded using `OneHotEncoder`.
   - Income was log-transformed to reduce skewness.
   - **SMOTE** was applied to address class imbalance in the target variable (mental health history).

3. **Model Selection**:
   - Several models were experimented with, including:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - LightGBM
   - The **Logistic Regression** model was selected for its balance between precision, recall, F1-score, and interpretability.

4. **Final Model**:
   - The final model was trained using `LogisticRegression` with hyperparameters tuned via grid search.
   - The final model achieved:
     - **Accuracy**: 61.72%
     - **Precision**: 39.19%
     - **Recall**: 45.86%
     - **F1 Score**: 42.26%

## Key Results

The final model’s key performance metrics are as follows:

| Metric      | Value   |
|-------------|---------|
| Accuracy    | 61.72%  |
| Precision   | 39.19%  |
| Recall      | 45.86%  |
| F1-Score    | 42.26%  |

The final model has a decent balance between precision and recall, and the results are explainable, making it a suitable choice for further evaluation.

## Installation

To run this project, you have two options to set up your environment: using `requirements.txt` with pip or `environment.yml` with Conda.

### Option 1: Using Conda

1. Clone the repo:
   ```bash
   git clone <repo-url>
   ```

2. Create and activate the environment using Conda:
    ```bash
    conda env create -f environment.yml
    conda activate axa-env
    ```

### Option 2: Using pip and requirements.txt

1. Clone the repo:
   ```bash
   git clone <repo-url>
   ```

2. Create and activate the environment using Conda:
    ```bash
    python -m venv axa-env
    source axa-env/bin/activate   # On Linux/MacOS
    axa-env\Scripts\activate      # On Windows
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. (Optional) If using Jupyter notebooks, ensure you have Jupyter installed:
    ```bash
    pip install jupyter
    ```

## Running the Code

1. Ensure the dataset is placed in the data/ folder.

2. Navigate to the src/ folder and run the Jupyter notebook titled EDA to run the EDA portion

3. Navigate to the src/ folder and run the Jupyter notebook titled Final_Model to build the final model for mental health prediction and to see the results

4. (Optional) For experiment notebooks, navigate to the test/ folder and run the Jupyter notebook.


## Model Limitations and Future Improvements

### Limitations

* The class imbalance in the dataset posed challenges, and while SMOTE helped, further tuning could enhance performance.
* The model could benefit from additional feature engineering and more complex models like XGBoost or deep learning techniques.

### Future Improvements

* Implement cross-validation techniques to enhance model generalizability.
* Consider using more sophisticated sampling methods or ensemble models for improved prediction.

## Contact
For any inquiries or questions, feel free to reach out.


---
