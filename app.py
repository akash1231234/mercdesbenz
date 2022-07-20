from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict, RandomizedSearchCV
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from train_model import *
import pickle
from numpy import nan
from sklearn.neighbors import KNeighborsRegressor
import xgboost
from sklearn.tree import DecisionTreeRegressor
from flask import Flask, render_template, request, redirect
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
app = Flask(__name__)


@app.route('/', methods=['GET'])
def Home():
    X0_variables = ['az', 't', 'w', 'y', 'x', 'f', 'ap', 'o', 'ay', 'al', 'h', 'z',
                    'aj', 'd', 'v', 'ak', 'ba', 'n', 'j', 's', 'af', 'ax', 'at', 'aq',
                    'av', 'm', 'k', 'a', 'e', 'ai', 'i', 'ag', 'b', 'am', 'aw', 'as',
                    'r', 'ao', 'u', 'l', 'c', 'ad', 'au', 'bc', 'g', 'an', 'ae', 'p',
                    'bb']
    X2_variables = ['n', 'ai', 'as', 'ae', 's', 'b', 'e', 'ak', 'm', 'a', 'aq', 'ag',
                    'r', 'k', 'aj', 'ay', 'ao', 'an', 'ac', 'af', 'ax', 'h', 'i', 'f',
                    'ap', 'p', 'au', 't', 'z', 'y', 'aw', 'd', 'at', 'g', 'am', 'j',
                    'x', 'ab', 'w', 'q', 'ah', 'ad', 'al', 'av', 'u']
    columns = ['X29', 'X54', 'X76', 'X127', 'X136', 'X162', 'X166', 'X178', 'X232',
               'X250', 'X261', 'X263', 'X272', 'X276', 'X279', 'X313', 'X314', 'X328']
    return render_template('index.html', X0_variable=sorted(X0_variables), X2_variable=sorted(X2_variables), column=columns)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        X0 = request.form['X0']
        X2 = request.form['X2']
        X29 = int(request.form['X29'])
        X54 = int(request.form['X54'])
        X76 = int(request.form['X76'])
        X127 = int(request.form['X127'])
        X136 = int(request.form['X136'])
        X162 = int(request.form['X162'])
        X166 = int(request.form['X166'])
        X178 = int(request.form['X178'])
        X232 = int(request.form['X232'])
        X250 = int(request.form['X250'])
        X261 = int(request.form['X261'])
        X263 = int(request.form['X263'])
        X272 = int(request.form['X272'])
        X276 = int(request.form['X276'])
        X279 = int(request.form['X279'])
        X313 = int(request.form['X313'])
        X314 = int(request.form['X314'])
        X328 = int(request.form['X328'])
        data = pd.DataFrame([[X0, X2, X29, X54, X76, X127, X136, X162, X166, X178, X232, X250, X261, X263, X272,
                              X276, X279, X313, X314, X328]], columns=['X0', 'X2', 'X29', 'X54', 'X76', 'X127', 'X136', 'X162', 'X166',
                                                                       'X178', 'X232', 'X250', 'X261', 'X263', 'X272', 'X276', 'X279', 'X313', 'X314', 'X328'])
        predictions = app_model(data)
        output = predictions[0]
        print(output)
        if output > 0:
            return render_template('predict.html', prediction_text=f'The amount of time taken on test bench is {round(output,2)} minutes')
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
