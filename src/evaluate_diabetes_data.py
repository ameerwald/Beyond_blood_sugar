# import packages 
import pandas as pd
import numpy as np
import os
# setting random seed for reproducability 
import random
random.seed(75)  # Python's built-in random module
np.random.seed(75)  # Numpy module
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import pickle as pkl
import warnings
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

# ignore all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# importing utility functions
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# importing functions from the utils scripts in the utils folder 
from utils.utils import preprocess_data, split_data, scale_data, initialize_models, train_models, evaluate_models, save_all_results
from utils.visualization_utils import plot_feature_importances, plot_model_performance, plot_recall_comparison, plot_recall_table, plot_coefficients, plot_coefficients_table

def main():
    # paths to files and directories 
    models_dir = 'models/'
    figure_dir = 'figures/'
    file_path = 'data/dataset2_cleaned.csv'
    output_dir = 'output/'

    # clean up the data 
    processed_data = preprocess_data(file_path)
    # split into train, val and test sets 
    X_train, X_test, y_train, y_test, feature_names = split_data(processed_data)
    # optional choice here to scale the data 
    X_train_std, X_test_std, scaler = scale_data(X_train, X_test)
    # initialize models
    models = initialize_models()
    # fit models 
    trained_models = train_models(models, X_train_std, y_train, models_dir)
     # evaluate on test set 
    model_results, best_params_results, feature_importances_results, coefficients_results = evaluate_models(trained_models, X_train_std, y_train, X_test_std, y_test, feature_names)
    # save results to different CSV files  
    save_all_results(model_results, best_params_results, feature_importances_results, output_dir)
    # save scaler 
    joblib.dump(scaler, 'output/scaler.pkl')
    # read CSV files for visualizing
    model_results_df = pd.read_csv('output/model_results.csv')
    best_params_table = pd.read_csv('output/best_params.csv')
    feature_importances_table = pd.read_csv('output/feature_importances.csv')
    # visualize and save tables (recall and coefficients)
    plot_recall_table(model_results_df, figure_dir)
    plot_coefficients_table(coefficients_results, figure_dir)
    #  make and save plots
    plot_feature_importances(feature_importances_table, figure_dir=figure_dir)
    plot_coefficients(coefficients_results, figure_dir)
    plot_model_performance(model_results_df, figure_dir=figure_dir)
    plot_recall_comparison(model_results_df, figure_dir=figure_dir)
    # print update 
    print("Model results have been saved to CSV files in the 'output' directory and visualizations to PNG images in the 'figures' directory")


if __name__=="__main__":
    main()

  


  