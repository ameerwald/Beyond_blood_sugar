# Beyond blood sugar: Machine learning attempts to predict diabetes

This repository is the product for the Data Science exam project, Spring 2024, by author Anja Feibel Meerwald.

## The data

These datasets are in the 'data' folder but can otherwise we found in the following links. 

Dataset2 is available [here](https://data.mendeley.com/datasets/jpp8bsjgrm/1) and is a cleaned up and refined version of [this](https://data.mendeley.com/datasets/wj9rwkp9c2/1) datset. 

Dataset1 can be found here [here](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt) which is the raw data from [here](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html) which I found via scikit-learn's website where information about the dataset can be found [here](https://scikit-learn.org/stable/datasets/toy_dataset.html).



# Repository 

| Folder         | Description          
| ------------- |:-------------:
| data   | This folder contains both datasets 
| figures  | Plots and tables of the results      
| output  |  Contains CSV files of results and metrics 
| models  | The saved models are here, folder is hidden due to size  
| src  | Py scripts 
| utils  | Functions used for the various py scripts        


## To run the scripts 

1. Clone the repository, either on ucloud or something like worker2.
2. From the command line, at the /Beyond_blood_sugar/ folder level, run the following lines of code. 

This will create a virtual environment and install the correct requirements.
``` 
bash setup.sh
```
While this runs the scripts and deactivates the virtual environment when it is done. 
```
bash run.sh
```

The scripts can also be run individually if that is preferred. 
This is done by running each script from the command line (see below) after the environment has been activated.
```
python3 src/evaluate_diabetes_data.py
python3 src/predict_new_diabetes_data.py
```

This has been tested on an ubuntu system on ucloud and therefore could have issues when run another way.