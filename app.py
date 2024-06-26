from flask import Flask, render_template, request, redirect, url_for
import zipfile
import os

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch
from scipy.linalg import norm
from statsmodels.tsa.stattools import acf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC

import io
import base64
import time
import seaborn as sns
import matplotlib.pyplot as plt

# Importing the functions from the activity_monitoring.py file
from activity_monitoring import rf_classifier,svm_classifier

app = Flask(__name__)

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods=['POST'])
def uploader():
    if 'file' not in request.files:
        return redirect(url_for('upload_file'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('upload_file'))
    
    if file:

        with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall("dataset") #file will be extracted in the dataset folder

        # Now we will train the model using the dataset

        with open('activity_monitoring.py', 'r') as f:
            a_py_content = f.read()

        exec(a_py_content)

        rf_accuracy,rf_f1,rf_conf_matrix,rf_graph_url = rf_classifier()
        svm_accuracy,svm_f1,svm_conf_matrix,svm_graph_url = svm_classifier()
    

        return render_template('result.html', rf_accuracy=rf_accuracy, rf_f1=rf_f1, rf_conf_matrix=rf_conf_matrix, rf_graph_url=rf_graph_url,svm_accuracy=svm_accuracy, svm_f1=svm_f1, svm_conf_matrix=svm_conf_matrix, svm_graph_url=svm_graph_url)

if __name__ == '__main__':
    app.run(debug=True)
