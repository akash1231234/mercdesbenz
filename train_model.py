import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_regression
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,cross_val_predict,RandomizedSearchCV
from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge,Lasso
from sklearn.metrics import r2_score,accuracy_score
from sklearn.tree import DecisionTreeRegressor
import xgboost
from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from numpy import nan
import pickle
dataset=pd.read_csv('mercedesbenz.csv')
def load_data(dataset):
    if 'y' in dataset.columns:
        X=dataset.drop(columns=['y','ID'])
        y=dataset['y']
        IQR=y.quantile(0.75)-y.quantile(0.25)
        upper_threshold=y.median()+3*IQR
        lower_threshold=y.median()-3*IQR
        y=pd.DataFrame(np.where(y>upper_threshold,upper_threshold,y),columns=['y'])
        return X,y
    elif 'ID' in dataset.columns:
        X=dataset.drop(columns=['ID'])
        return X
    else:
        X=dataset.copy()
        return X

minmax=MinMaxScaler(feature_range=(0,1))
scaler=StandardScaler()
le=LabelEncoder()
oe=OrdinalEncoder()
ohe=OneHotEncoder(sparse=False,drop='first')
def preprocessing(datset,X):
    global num_removed_cols
    num_removed_cols=[]
    global cat_encoded_cols
    cat_encoded_cols=[]
    global cat_cols
    global num_cols
    cat_cols=X.select_dtypes(include='object').columns.values.tolist()
    num_cols=X.select_dtypes(exclude='object').columns.values.tolist()
    for col in num_cols:
        if X[col].nunique()==1:
            num_removed_cols.append(col)
    for col in cat_cols:
        Mean_encoded_col=dataset.groupby(col)['y'].median().sort_values(ascending=True).index
        encoded_col={k:i for i,k in enumerate(Mean_encoded_col,0)}
        encoded_dict={col:encoded_col}
        cat_encoded_cols.append(encoded_dict)
    return num_removed_cols,cat_encoded_cols

def transformation(X):
    for col in num_removed_cols:
        X=X.drop(labels=col,axis=1)
    for cols in cat_encoded_cols:
        for key,values in cols.items():
            for col in cat_cols:
                if key==col:
                    X[col]=X[col].map(values)
                    X[[col]]=scaler.fit_transform(X[[col]])
    return X

def feature_selection(X,y):
    Selector=SelectKBest(score_func=f_regression,k=20)
    Selector.fit(X,y)
    X_new=Selector.fit_transform(X,y)
    X_new=pd.DataFrame(Selector.fit_transform(X,y),columns=X.columns[Selector.get_support()])
    global selected_columns
    selected_columns=X.columns[Selector.get_support()]
    return X_new


models={'Linear Regression':LinearRegression(),'Ridge Regression':Ridge(),'Lasso Regression':Lasso(),'Decision Tree':DecisionTreeRegressor(),
       'XGBoost':xgboost.XGBRegressor(),'AdaBoost':AdaBoostRegressor(),'Bagging Regressor':BaggingRegressor(),'Gradient Boosting':GradientBoostingRegressor(),
       'Random Forest': RandomForestRegressor()}
def train_model(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    for model_names,model in models.items():
        if model_names=='XGBoost' :
            params={
                     "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
                     "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
                     "min_child_weight" : [ 1, 3, 5, 7 ],
                     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
                     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ], 
                    }
            xgboost_test=RandomizedSearchCV(model,param_distributions=params,n_iter=15,\
                                 scoring='r2',n_jobs=-1,cv=5,verbose=8,random_state=42)
            # estimator=xgboost_test.best_estimator_
            # regressor=
            xgboost_model=xgboost_test.fit(X_train,y_train)
            xgboost_train=xgboost_model.predict(X_train)
            xgboost_pred=xgboost_model.predict(X_test)
            print('The accuracy for train is',r2_score(y_train,xgboost_train))
            print('The accuracy for test is',r2_score(y_test,xgboost_pred))
            pickle.dump(xgboost_test,open('XGBoost_regression.pkl','wb'))
            predictions=pd.DataFrame()
            predictions['y_train_true']=y_train
            predictions['y_train_pred']=xgboost_train.tolist()
            test_predictions=pd.DataFrame()
            test_predictions['y_test_true']=y_test
            test_predictions['y_test_pred']=xgboost_pred.tolist()
            return predictions,test_predictions
        else:
            continue
def modelling():
    global y
    dataset=pd.read_csv('mercedesbenz.csv')
    X,y=load_data(dataset)
    preprocessing(dataset,X)
    X=transformation(X)
    X=feature_selection(X,y)
    # predictions,test_predictions=train_model(X,y)
    return X
def test_model(data):
    modelling()
    X=load_data(data)
    X=transformation(X)
    X=X[selected_columns]
    xgboost_model=pickle.load(open('XGBoost_regression.pkl','rb'))
    predictions=pd.DataFrame()
    predictions['y']=xgboost_model.predict(X)
    return predictions

def app_transformation(X):
    for cols in cat_encoded_cols:
        for key,values in cols.items():
            if key in X.columns:
                X[key]=X[key].map(values)
                X[[key]]=scaler.transform(X[[key]])
            else:
                continue
    return X

def app_model(data):
    X=modelling()
    X=app_transformation(data)
    xgboost_model=pickle.load(open('XGBoost_regression.pkl','rb'))
    predictions=xgboost_model.predict(X)
    return predictions