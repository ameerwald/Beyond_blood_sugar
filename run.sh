#!/usr/bin/env bash

#activate virtual environment
source ./env/bin/activate

# run the code
python3 src/evaluate_diabetes_data.py
python3 src/predict_new_diabetes_data.py

# deactive the venv
deactivate