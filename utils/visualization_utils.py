# import packages 
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


###########################################################################
##################### original dataset plotting ###########################
###########################################################################

def plot_feature_importances(feature_importances_table, figure_dir, filename_prefix='feature_importances'):
    """
    Plot and save the feature importances as separate bar plots for each model.
    
    Parameters:
    - feature_importances_table (pd.DataFrame): DataFrame containing feature importances with columns 'Feature', 'Importance', and 'Model'.
    - figure_dir (str): Directory to save the figures.
    - filename_prefix (str): Prefix for the filename to save the figure.
    """
    # checking if output directory exists and making one if not
    os.makedirs(figure_dir, exist_ok=True)
    # using 'Model' column to distinguish between models
    models = feature_importances_table['Model'].unique()
    for model in models:
        model_data = feature_importances_table[feature_importances_table['Model'] == model]
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Feature', y='Importance', data=model_data, color=sns.color_palette()[0])
        plt.title(f'Feature Importances for {model}')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        # rotate feature names so it looks better 
        plt.xticks(rotation=45)  
        # save the plot as a PNG file
        plt.savefig(os.path.join(figure_dir, f'{filename_prefix}_{model}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_model_performance(model_results_df, figure_dir):
    """
    Plot and save the model performance on training and test sets.
    
    Parameters:
    - model_results_df (pd.DataFrame): DataFrame containing model evaluation metrics.
    - figure_dir (str): Directory to save the figures.
    """
    # checking if output directory exists and making one if not
    os.makedirs(figure_dir, exist_ok=True)
    # getting them to stack on top of each other in two rows 
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
    # plot on the first graph, first axis
    model_results_df.set_index('Model')[['Train Precision', 'Train Recall', 'Train F1-Score']].plot(kind='bar', ax=axes[0])
    axes[0].set_ylabel('Score')
    axes[0].set_title('Model Performance on Training Set')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    # adjusting x-ticks to line them up correctly
    train_models = model_results_df['Model']
    axes[0].set_xticks(range(len(train_models)))
    axes[0].set_xticklabels(train_models, rotation=45, ha='right')
    # plot on the second graph, second axis 
    model_results_df.set_index('Model')[['Test Precision', 'Test Recall', 'Test F1-Score']].plot(kind='bar', ax=axes[1])
    axes[1].set_ylabel('Score')
    axes[1].set_title('Model Performance on Test Set')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    # adjusting x-ticks to line them up correctly
    test_models = model_results_df['Model']
    axes[1].set_xticks(range(len(test_models)))
    axes[1].set_xticklabels(test_models, rotation=45, ha='right')
    plt.tight_layout()
    # save plot
    plt.savefig(os.path.join(figure_dir, 'classification_metrics_trainandtest.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_coefficients_table(coefficients_results, figure_dir):
    """
    Create a visual representation of the coefficients and save it as a figure.
    
    Parameters:
    - coefficients_results (list of pd.DataFrame): List containing DataFrames of coefficients from different models.
    - figure_dir (str): Directory to save the figures.
    """
    if not coefficients_results:
        print("No coefficients to display.")
        return

    # checking if output directory exists and making one if not
    os.makedirs(figure_dir, exist_ok=True)

    # concatenate all dfs into one df
    full_coefficients_df = pd.concat(coefficients_results, ignore_index=True)

    # create a figure and an axes object for displaying the table
    fig, ax = plt.subplots(figsize=(12, len(full_coefficients_df) * 0.4)) 
    ax.axis('tight')
    ax.axis('off')

    # prep data for the table
    columns = ['Model', 'Feature', 'Coefficient']
    cell_text = full_coefficients_df[columns].values
    column_labels = columns

    # create the table
    table = ax.table(cellText=cell_text, colLabels=column_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)  
    table.scale(1, 1.5) 

    # save the figure
    plt.savefig(os.path.join(figure_dir, 'coefficients_table.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_coefficients(coefficients_results, figure_dir):
    # checking if output directory exists and making one if not
    os.makedirs(figure_dir, exist_ok=True)

    # concatenate all df objects in the list
    if isinstance(coefficients_results, list):
        coefficients_table = pd.concat(coefficients_results, ignore_index=True)
    else:
        coefficients_table = coefficients_results

    models = coefficients_table['Model'].unique()

    for model in models:
        model_data = coefficients_table[coefficients_table['Model'] == model]
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Feature', y='Coefficient', data=model_data)
        plt.title(f'Coefficients for {model}')
        plt.xlabel('Features')
        plt.ylabel('Coefficient Value')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(figure_dir, f'coefficients_{model}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_recall_comparison(model_results_df, figure_dir):
    """
    Plot and save the recall comparison between training and test sets.
    
    Parameters:
    - model_results_df (pd.DataFrame): DataFrame containing model evaluation metrics.
    - figure_dir (str): Directory to save the figures.
    """
    # checking if output directory exists and making one if not
    os.makedirs(figure_dir, exist_ok=True)

    # get train and test recall values 
    recall_df = model_results_df[['Model', 'Train Recall', 'Test Recall']].copy()

    # pivot for better plotting
    recall_pivot = recall_df.set_index('Model')

    # plotting the comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    recall_pivot.plot(kind='bar', ax=ax, color=['#C1E1C1', '#00008B'],edgecolor='grey')
    ax.set_ylabel('Recall Score')
    ax.set_title('Recall Metric Comparison on Training and Test Sets')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # adjusting x-ticks so they line up correctly
    ax.set_xticks(range(len(recall_pivot)))
    ax.set_xticklabels(recall_pivot.index, rotation=45, ha='right')
    plt.legend(title='Dataset')
    plt.tight_layout()

    # save plot
    plt.savefig(os.path.join(figure_dir, 'recall_trainandtest.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_recall_table(model_results_df, figure_dir):
    """
    Create a visual representation of the recall values and save it as a figure.
    
    Parameters:
    - model_results_df (pd.DataFrame): DataFrame containing model evaluation metrics.
    - figure_dir (str): Directory to save the figures.
    """
    # checking if output directory exists and making one if not
    os.makedirs(figure_dir, exist_ok=True)

    # get recall values
    recall_df = model_results_df[['Model', 'Train Recall', 'Test Recall']].set_index('Model')
    
    # create a figure and an axes object
    fig, ax = plt.subplots(figsize=(8, 3)) 
    ax.axis('tight')
    ax.axis('off')

    # create the table
    table = ax.table(cellText=recall_df.values, colLabels=recall_df.columns, rowLabels=recall_df.index, cellLoc = 'center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)  
    table.scale(1.2, 1.2)  

    # save the figure
    plt.savefig(os.path.join(figure_dir, 'recall_table_figure.png'), dpi=300, bbox_inches='tight')
    plt.close()




###########################################################################
####################### new dataset plotting ##############################
###########################################################################



def plot_new_data_metrics(metrics_df, figure_dir='figures/'):
    """
    Plot and save the model performance metrics for the new dataset.
    
    Parameters:
    - metrics_df (pd.DataFrame): DataFrame containing the model performance metrics for the new dataset.
    - figure_dir (str): Directory to save the figures.
    """
    # checking if output directory exists and making one if not
    os.makedirs(figure_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_df.set_index('Model').plot(kind='bar', ax=ax)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance on New Dataset (Precision, Recall, F1-score)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # adjusting x-ticks to they line up correctly
    ax.set_xticks(range(len(metrics_df)))
    ax.set_xticklabels(metrics_df['Model'], rotation=45, ha='right')
    plt.legend(title='Metrics')
    plt.tight_layout()

    # save the figure 
    plt.savefig(os.path.join(figure_dir, 'new_data_metrics.png'), dpi=300, bbox_inches='tight')
    plt.show()



def plot_new_recall_comparison(test_metrics_df, new_data_metrics_df, figure_dir='figures/'):
    """
    Plot and save the recall comparison between the test set and the new dataset.
    
    Parameters:
    - test_metrics_df (pd.DataFrame): DataFrame containing the model performance metrics for the test set.
    - new_data_metrics_df (pd.DataFrame): DataFrame containing the model performance metrics for the new dataset.
    - figure_dir (str): Directory to save the figures.
    """
    # checking if output directory exists and making one if not
    os.makedirs(figure_dir, exist_ok=True)

    # taking only the 'model' and 'recall' columns for comparison
    test_recall = test_metrics_df[['Model', 'Test Recall']].copy()  
    test_recall.rename(columns={'Test Recall': 'Recall'}, inplace=True)
    test_recall['Dataset'] = 'Test Set'

    new_dataset_recall = new_data_metrics_df[['Model', 'Recall']].copy()
    new_dataset_recall['Dataset'] = 'New Dataset'

    # combining the recall data from both 
    combined_recall_data = pd.concat([test_recall, new_dataset_recall])

    # pivoting the data to plot 
    recall_pivot = combined_recall_data.pivot(index='Model', columns='Dataset', values='Recall')

    # reindexing to ensure consistent order of model names
    model_order = test_metrics_df['Model'].unique()
    recall_pivot = recall_pivot.reindex(model_order)


    # plotting the recall comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    recall_pivot.plot(kind='bar', ax=ax, color=['#6495ED', '#00008B'], edgecolor='grey')
    ax.set_ylabel('Recall Score')
    ax.set_title('Recall Metric Comparison on Test Set and New Dataset')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # adjusting x-ticks 
    ax.set_xticks(range(len(recall_pivot)))
    ax.set_xticklabels(recall_pivot.index, rotation=45, ha='right')
    plt.legend(title='Dataset')
    plt.tight_layout()

    # save the figure 
    plt.savefig(os.path.join(figure_dir, 'recall_comparison_both_datasets.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_new_recall_comparison(test_metrics_df, new_data_metrics_df, figure_dir='figures/'):
    """
    Plot and save the recall comparison between the test set and the new dataset.

    Parameters:
    - test_metrics_df (pd.DataFrame): DataFrame containing the model performance metrics for the test set.
    - new_data_metrics_df (pd.DataFrame): DataFrame containing the model performance metrics for the new dataset.
    - figure_dir (str): Directory to save the figures.
    """
    # checking if output directory exists and making one if not
    os.makedirs(figure_dir, exist_ok=True)

    # taking only the 'model' and 'recall' columns for comparison
    test_recall = test_metrics_df[['Model', 'Test Recall']].copy()
    test_recall.rename(columns={'Test Recall': 'Recall'}, inplace=True)
    test_recall.loc[:, 'Dataset'] = 'Test Set'

    new_dataset_recall = new_data_metrics_df[['Model', 'Recall']].copy()
    new_dataset_recall.loc[:, 'Dataset'] = 'New Dataset'

    # combining the recall data from both 
    combined_recall_data = pd.concat([test_recall, new_dataset_recall])

    # pivoting the data to plot 
    recall_pivot = combined_recall_data.pivot(index='Model', columns='Dataset', values='Recall')

    # reindexing to ensure consistent order of model names
    model_order = test_metrics_df['Model'].unique()
    recall_pivot = recall_pivot.reindex(model_order)

    # plotting the recall comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    recall_pivot.plot(kind='bar', ax=ax, color=['#6495ED', '#00008B'], edgecolor='grey')
    ax.set_ylabel('Recall Score')
    ax.set_title('Recall Metric Comparison on Test Set and New Dataset')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # adjusting x-ticks 
    ax.set_xticks(range(len(recall_pivot)))
    ax.set_xticklabels(recall_pivot.index, rotation=45, ha='right')
    plt.legend(title='Dataset')
    plt.tight_layout()

    # save the figure 
    plt.savefig(os.path.join(figure_dir, 'recall_comparison_both_datasets.png'), dpi=300, bbox_inches='tight')
    plt.show()

