import os
import pandas as pd
import numpy as np
import joblib
import warnings
# setting random seed for reproducability 
import random
random.seed(75)  # Python's built-in random module
np.random.seed(75)  # Numpy module
from sklearn.metrics import classification_report

# ignore all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# importing utility functions
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.utils import preprocess_new_data, scale_new_data, evaluate_models_on_new_data
from utils.visualization_utils import plot_new_data_metrics, plot_new_recall_comparison, plot_new_recall_comparison


def main():
    # paths to files and directories
    new_data_file_path = 'data/raw_dataset1.txt'
    models_dir = 'models/'
    figure_dir = 'figures/'
    scaler_path = 'output/scaler.pkl'
    test_metrics_file_path = 'output/model_results.csv'
    output_dir = 'output/'
    true_label = 1

    # preprocess the new data
    X_new, feature_names = preprocess_new_data(new_data_file_path)
    # scale the new data - with the same scaler 
    X_new_std = scale_new_data(X_new, scaler_path)
    # evaluate new data with loaded models
    model_results, feature_importances_results = evaluate_models_on_new_data(models_dir, X_new_std, true_label, output_dir, feature_names)
    # make results into dataframe
    new_data_metrics_df = pd.DataFrame(model_results)
    all_feature_importances_df = pd.concat(feature_importances_results, ignore_index=True)
    # save new data metrics to CSV
    new_data_metrics_df.to_csv(os.path.join(output_dir, 'new_data_metrics.csv'), index=False)
    # read the saved CSV files 
    test_metrics_df = pd.read_csv(test_metrics_file_path)
    new_data_metrics_df = pd.read_csv(os.path.join(output_dir, 'new_data_metrics.csv'))
    # visualize and save recall table as a figure
    plot_new_recall_comparison(test_metrics_df, new_data_metrics_df, figure_dir='figures/')
    # plot and save visualizations
    plot_new_data_metrics(new_data_metrics_df, figure_dir=figure_dir)
    plot_new_recall_comparison(test_metrics_df, new_data_metrics_df, figure_dir=figure_dir)
    # print update 
    print("New metrics have been saved to CSV files in the 'output' directory and visualizations to PNG images in the 'figures' directory.")

if __name__ == "__main__":
    main()
