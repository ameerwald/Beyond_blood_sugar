# import packages 
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import pickle as pkl
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

########################################################################################
########################## First dataset functions #######################################
#######################################################################################

def preprocess_data(file_path):
    """
    Load and preprocess the data from a given file path.

    Parameters:
    - file_path (str): The path to the CSV file with the data.

    Returns:
    - pd.DataFrame: The preprocessed DataFrame.
    """
    # loading data
    data = pd.read_csv(file_path)

    # rename columns to match other dataset 
    data.rename(columns={'Gender': 'sex', 'AGE': 'age', 'BMI': 'bmi'}, inplace=True)

    # remove all 1 values which represents predicted diabetics
    data = data[data['Class'] != 1]

    # rename the 'Class' values from 2 to 1 (so 1 = diabetics and 0 = nondiabetics)
    data['Class'] = data['Class'].replace({2: 1})

    return data


def split_data(data, test_size=0.15, random_state=35):
    """
    Split the data into training, validation, and test sets.

    Parameters:
    - data (pd.DataFrame): The preprocessed DataFrame.
    - test_size (float): Proportion of the data to be used as the test set.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - tuple: Split data as (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # picking features that overlap with other dataset 
    X = data[['age', 'sex', 'bmi', 'Chol', 'TG', 'HDL', 'LDL']]
    y = data['Class']

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # get feature names here
    feature_names = X.columns  

    return X_train, X_test, y_train, y_test, feature_names 



def scale_data(X_train, X_test):
    """
    Standardize the data.

    Parameters:
    - X_train (pd.DataFrame): Training feature set.
    - X_test (pd.DataFrame): Test feature set.

    Returns:
    - tuple: Standardized data as (X_train_std,  X_test_std).
    """
    # loading the scaler 
    scaler = StandardScaler()

    # scaling the variables 
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    return X_train_std, X_test_std, scaler


def initialize_models():
    """
    Initialize the models dictionary including non-grid models.

    Returns:
    - dict: Dictionary containing model definitions and hyperparameters.
    """
    return {
        'Dummy': {
            'model': DummyClassifier(strategy='uniform', random_state=15), 
            'params': None},
        'Naive Bayes': {
            'model': GaussianNB(), 
            'params': None},
        'KNN': {
            'model': KNeighborsClassifier(), 
            'params': {'n_neighbors': [3, 5, 7, 9]}},
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, class_weight='balanced'), 
            'params': {'C': np.logspace(-4, 4, 10), 'solver': ['liblinear', 'lbfgs', 'saga']}},
        'Decision Tree': {
            'model': DecisionTreeClassifier(), 
            'params': {'max_depth': [3, 5, 10, 20], 'min_samples_split': [2, 5, 10]}},
        'Random Forest': {
            'model': RandomForestClassifier(random_state=75), 
            'params': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20], 'min_samples_split': [2, 5, 10]}},
        'XGBoost': {
            'model': XGBClassifier(eval_metric='logloss', random_state=75), 
            'params': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15], 'learning_rate': [0.01, 0.1, 0.2]}},
    }


def train_models(models, X_train_std, y_train, models_dir):
    """
    Train and save models using the training data.
    
    Parameters:
    - models (dict): Dictionary containing model definitions and hyperparameters.
    - X_train_std (pd.DataFrame): Standardized training data.
    - y_train (pd.Series): Training target data.
    - models_dir (str): Directory to save the models.
    
    Returns:
    - dict: Dictionary containing trained models and their parameters.
    """
    # checking if output directory exists and making one if not
    os.makedirs(models_dir, exist_ok=True) 
    trained_models = {}
    for name, data in models.items():

        # parameter tuning with GridSearchCV
        if data['params']:  
            grid_search = GridSearchCV(data['model'], data['params'], cv=5, scoring='recall')
            grid_search.fit(X_train_std, y_train)
            model = grid_search.best_estimator_
            params = grid_search.best_params_
        else:
            model = data['model']
            model.fit(X_train_std, y_train)
            params = None

        model_path = os.path.join(models_dir, f"{name.replace(' ', '_').lower()}_best_model.pkl")

        #save the trained model
        joblib.dump(model, model_path) 
        trained_models[name] = {'model': model, 'params': params}

    return trained_models


def evaluate_models(trained_models, X_train_std, y_train, X_test_std, y_test, feature_names):
    """
    Evaluate trained models on both training and test sets and collect metrics,
    including feature importances and best parameters.
    
    Parameters:
    - trained_models (dict): Dictionary of trained models and their parameters.
    - X_train_std (pd.DataFrame): Standardized training data.
    - y_train (pd.Series): Training target data.
    - X_test_std (pd.DataFrame): Standardized test data.
    - y_test (pd.Series): Test target data.
    - feature_names (list): List of feature names used in models that support feature importance.
    
    Returns:
    - tuple: Contains model results, best parameters, and feature importances.
    """
    model_results = []
    best_params_results = []
    feature_importances_results = []
    coefficients_results = []
    
    for name, info in trained_models.items():
        model = info['model']
        params = info.get('params', None)

        # evaluate on training set
        y_train_pred = model.predict(X_train_std)
        train_metrics = classification_report(y_train, y_train_pred, output_dict=True, zero_division=0)['1']

        # evaluate on test set
        y_test_pred = model.predict(X_test_std)
        test_metrics = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)['1']

        # collect results
        model_results.append({
            'Model': name,
            'Train Precision': train_metrics['precision'],
            'Train Recall': train_metrics['recall'],
            'Train F1-Score': train_metrics['f1-score'],
            'Test Precision': test_metrics['precision'],
            'Test Recall': test_metrics['recall'],
            'Test F1-Score': test_metrics['f1-score']
        })

        # collect best parameters
        if params:
            best_params_results.append({
                'Model': name,
                'Best Parameters': params
            })

        # collect feature importances if available
        if hasattr(model, 'feature_importances_'):
            feature_importances = {
                'Feature': feature_names,
                'Importance': model.feature_importances_,
                'Model': name
            }
            feature_importances_results.append(pd.DataFrame(feature_importances))

        # collect coefficients for LR models
        if hasattr(model, 'coef_'):
            coefficients = {
                'Feature': feature_names,
                'Coefficient': model.coef_[0],
                'Model': name
            }
            coefficients_results.append(pd.DataFrame(coefficients))

    return model_results, best_params_results, feature_importances_results, coefficients_results



def save_all_results(model_results, best_params_results, feature_importances_results, output_dir = 'output/'):
    """
    Save the model results, best parameters, feature importances to CSV files.

    Parameters:
    - model_results (list or pd.DataFrame): Combined model evaluation results for both training and test data.
    - best_params_results (list): List of dictionaries containing best parameters for each model.
    - feature_importances_results (list): List of DataFrames containing feature importances for each model.
    - output_dir (str): Directory to save the output CSV files.
    """
    # checking if output directory exists and making one if not
    os.makedirs(output_dir, exist_ok=True)  

    # convert list of dictionaries to df if necessary and save
    if not isinstance(model_results, pd.DataFrame):
        model_results_df = pd.DataFrame(model_results)
    else:
        model_results_df = model_results
    model_results_df.to_csv(os.path.join(output_dir, 'model_results.csv'), index=False)

    if isinstance(best_params_results, list):
        best_params_df = pd.DataFrame(best_params_results)
    else:
        best_params_df = best_params_results
    best_params_df.to_csv(os.path.join(output_dir, 'best_params.csv'), index=False)

    if feature_importances_results and isinstance(feature_importances_results[0], pd.DataFrame):
        all_feature_importances_df = pd.concat(feature_importances_results, ignore_index=True)
        all_feature_importances_df.to_csv(os.path.join(output_dir, 'feature_importances.csv'), index=False)






########################################################################################
########################## Second dataset functions #######################################
#######################################################################################

def preprocess_new_data(file_path):
    """
    Load and preprocess the new data.

    Parameters:
    - file_path (str): Path to the new data file.

    Returns:
    - pd.DataFrame: Preprocessed feature DataFrame.
    """
    # load raw data
    new_data = pd.read_csv(file_path, sep='\s+')

    # remove Obs column which just seems to be an index
    new_data = new_data.drop('Obs', axis=1)

    # renaming the s1-6 columns so they make some sense 
    new_data.rename(columns={'s1': 'TC', 's2': 'LDL', 's3': 'HDL', 's4': 'Chol', 's5': 'TG', 's6': 'Glu', 'y': 'target'}, inplace=True)
    
    # changing the sex values so it's 0 or 1 
    new_data['sex'] = new_data['sex'] - 1
    
    # picking features that overlap with other dataset
    X_new = new_data[['age', 'sex', 'bmi', 'Chol', 'TG', 'HDL', 'LDL']]
    
    # get feature names here to use later 
    feature_names = X_new.columns 

    return X_new, feature_names


def scale_new_data(X_new, scaler_path):
    """
    Scale the new data using the provided scaler.

    Parameters:
    - X_new (pd.DataFrame): The new data to be scaled.
    - scaler_path (str): Path to the saved scaler file.

    Returns:
    - np.ndarray: Scaled new data.
    """
    # load the saved scaler
    scaler = joblib.load(scaler_path)

    # scale the new data
    X_new_std = scaler.transform(X_new)

    return X_new_std
    


def evaluate_models_on_new_data(models_dir, X_new_std, true_label, output_dir, feature_names):
    """
    Load models, evaluate them on new data, and save the results to a CSV file.
    
    Parameters:
    - models_dir (str): Directory where the models are saved.
    - X_new_std (pd.DataFrame): Standardized new data.
    - true_label (int): The true label for the new data.
    - output_dir (str): Directory to save the output CSV file.
    - feature_names (list): List of feature names used in models that support feature importance.
    
    Returns:
    - tuple: Contains model results and feature importances.
    """
    model_results = []
    feature_importances_results = []
    
    # loading the previously saved models 
    for model_file in os.listdir(models_dir):
        if model_file.endswith('_best_model.pkl'):
            model_path = os.path.join(models_dir, model_file)
            best_model = joblib.load(model_path)
            model_name = model_file.replace('_best_model.pkl', '').replace('_', ' ').title()
            
            # fixing names for KNN and XGBoost so they don't get plotted weird 
            if model_name.lower() == 'knn':
                model_name = 'KNN'
            elif model_name.lower() == 'xgboost':
                model_name = 'XGBoost'
            
            # evaluate on new data 
            y_new_pred = best_model.predict(X_new_std)
            true_labels = [true_label] * len(y_new_pred)
            report = classification_report(true_labels, y_new_pred, labels=[true_label], output_dict=True, zero_division=0)
            positive_metrics = report[str(true_label)]
            
            # collect the results 
            model_results.append({
                'Model': model_name,
                'Best Model': best_model,
                'Precision': positive_metrics['precision'],
                'Recall': positive_metrics['recall'],
                'F1-Score': positive_metrics['f1-score']
            })
            
            # collect feature importances if available
            if hasattr(best_model, 'feature_importances_'):
                feature_importances = {
                    'Feature': feature_names,
                    'Importance': best_model.feature_importances_,
                    'Model': model_name
                }
                feature_importances_results.append(pd.DataFrame(feature_importances))

    return model_results, feature_importances_results
