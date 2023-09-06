import pathlib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score,  recall_score, f1_score, roc_auc_score, accuracy_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
# import packages for hyperparameters tuning
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from optuna.visualization.matplotlib import plot_optimization_history

from sklearn.utils import class_weight
from sklearn.model_selection import cross_validate
import pickle
from hyperparameter_base import objective, tuning
from sklearn.linear_model import LogisticRegression
import scikitplot as skplt
import time
from xgboost import plot_importance
import seaborn as sns
import copy
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.io as pio
pio.renderers.default='browser'

#import warnings
#warnings.filterwarnings('ignore')


# current working directory
path = pathlib.Path().absolute()

################
### Folders ####
################

data_input_folder   = "{}\output".format(path.__str__().replace("base_models", "dataset"))
output_folder       = "{}\\output".format(path)

################
# Load Dataset #
################

# Data for Pop Vs Other Classification
df_train = pd.read_excel("{}\\genres_bi.xlsx".format(data_input_folder), sheet_name = 0)
df_test = pd.read_excel("{}\\genres_bi.xlsx".format(data_input_folder), sheet_name = 1)

X_train = df_train.drop(["Unnamed: 0", "id", "artist name", "track name", "class"], axis = 1)
y_train = df_train["class"]

X_test = df_test.drop(["Unnamed: 0", "id", "artist name", "track name", "class"], axis = 1)
y_test = df_test["class"]


############################
###### Tune ML Models ######
############################

# Define list of models used for prediction
models = ["logit", "knn", "svm", "lda", "qda", "dt", "rf", "gbt"] 

# Init dict to store tuning parameters
best_params_base = {}

# Tune models for each lead-type
tuner = objective(X_train, y_train) #define objective 
best_params_base, timer = tuning(tuner, models) # store

## Save best parameters
with open("{}\\best_params_base.pkl".format(output_folder), "wb") as fp:
    pickle.dump(best_params_base, fp)    
     
#######################################
#### Prediction on test data ####
#######################################

###### Initialize models
# init dictionary
models_dict_base = {}

# Needed later for model stacking, but easier to setup here
rf_yhat, gbt_yhat, logit_yhat, dt_yhat, svm_yhat, lda_yhat, qda_yhat, knn_yhat = [], [], [], [], [], [], [], []

#initiate models and store in dict, use optuna parameters
gbt = XGBClassifier(objective="binary:logistic", **best_params_base["gbt"], eval_metric="auc", random_state = 123, n_jobs = -1)
rf = RandomForestClassifier(**best_params_base["rf"], random_state = 123, n_jobs = -1)
logit = LogisticRegression(**best_params_base["logit"], penalty = "elasticnet", solver = "saga", max_iter = 10000)        
dt = DecisionTreeClassifier(**best_params_base["dt"], random_state = 123)
knn = KNeighborsClassifier(**best_params_base["knn"], n_jobs = -1)
svm = SVC(**best_params_base["svm"], probability = True)
qda = QuadraticDiscriminantAnalysis(**best_params_base["qda"])
lda = LinearDiscriminantAnalysis(**best_params_base["lda"], solver = "lsqr")

models_dict_base = {"logit": [logit, logit_yhat],
                    "knn": [knn, knn_yhat],
                    "svm": [svm, svm_yhat],
                    "lda": [lda, lda_yhat],
                    "qda" : [qda, qda_yhat],
                    "dt": [dt, dt_yhat],
                    "rf" : [rf, rf_yhat], 
                    "gbt" : [gbt, gbt_yhat]
                    }

# make deepcopy for later, before fitted to data     
models_dict_stack = copy.deepcopy(models_dict_base)


##########################    
### Optimal Thresholds ###
##########################
with open("{}\\models_dict_base.pkl".format(output_folder), "rb") as fp:
        models_dict_base = pickle.load(fp) 
# Thresholds are chosen by maximizing F1-Score 
# Use validation set to avoid data leakage on test set
X1, X2, y1, y2 = train_test_split(X_train, y_train, test_size = 0.2, random_state=123)        

#init dicts
base_thresholds = {}
proba_thresh = {}
prc_thresh = {}

for mod in models_dict_base.keys():
    models_dict_base[mod][0].fit(X1, y1) #fit model to training data
    proba_thresh[mod] = models_dict_base[mod][0].predict_proba(X2) # predict probabilities on validation data
    
    pre, rec, thresholds_prc = precision_recall_curve(y2, proba_thresh[mod][:, 1]) #calculate precision recall metrics
    prc_thresh[mod] = [pre, rec, thresholds_prc] #save metrics
    numer = (2 * prc_thresh[mod][0] * prc_thresh[mod][1])
    denom = (prc_thresh[mod][0]  + prc_thresh[mod][1] )
    fscore = np.divide(numer, denom, out=np.zeros_like(denom), where=(denom!=0)) #denominator of F1-Score can be zero, hence replace nan with 0 in fscroe
    idx = np.argmax(fscore) #locate largest F1 Score
    base_thresholds[mod] = prc_thresh[mod][2][idx] #save threshild that produces largest F1 Score
 

####################################
### Final prediction on test set ###
####################################

###### Calculate evaluation metrics for all models ######
# init dicts for storage
base_preds = {}
base_proba = {}
metrics_base = []
rocauc = {}
prc = {}
 
 
# predict on validation data
  
for mod in models_dict_base.keys():
    models_dict_base[mod][0].fit(X_train, y_train) #fit model to training data
    base_proba[mod] = models_dict_base[mod][0].predict_proba(X_test)[:, 1] # predict probabilities
    base_preds[mod] = copy.deepcopy(base_proba[mod])
    
    base_preds[mod][base_preds[mod]>=base_thresholds[mod]] = 1 #make predictions based on thresholds
    base_preds[mod][base_preds[mod]<base_thresholds[mod]] = 0
    
    #base_preds[mod] = models_dict_base[mod][0].predict(X_test) #not using optimized thresholds
    
    #calculate metrics
    acc = np.round(accuracy_score(y_test, base_preds[mod]), 4)
    precision = np.round(precision_score(y_test, base_preds[mod], average= 'binary'), 4)
    recall = np.round(recall_score(y_test, base_preds[mod], average= 'binary'), 4)
    f1 = np.round(f1_score(y_test, base_preds[mod], average= 'binary'), 4)
    auc = np.round(roc_auc_score(y_test, base_proba[mod]), 3)
    fpr, tpr, thresholds_roc = roc_curve(y_test, base_proba[mod])
    pre, rec, thresholds_prc = precision_recall_curve(y_test, base_proba[mod])
    
    rocauc[mod] = [fpr, tpr, thresholds_roc, auc]
    prc[mod] = [pre, rec, thresholds_prc]
    
    temp = [mod, timer[mod], acc, auc, precision, recall, f1] #gather metrics
    metrics_base.append(temp)

base_results = pd.DataFrame(metrics_base, columns= ["Model", "Time", "Accuracy", "ROC-AUC", "Precision","Recall", "F1"])
base_results.sort_values(by= ['F1'], ascending = False, inplace= True)
base_results.reset_index(drop = True, inplace=True)

## Save model_dict, predictions and results
with open("{}\\models_dict_base.pkl".format(output_folder), "wb") as fp:
    pickle.dump(models_dict_base, fp)
with open("{}\\models_dict_stack.pkl".format(output_folder), "wb") as fp:
    pickle.dump(models_dict_stack, fp)    
with open("{}\\base_preds.pkl".format(output_folder), "wb") as fp:
    pickle.dump(base_preds, fp)
with open("{}\\base_proba.pkl".format(output_folder), "wb") as fp:
    pickle.dump(base_proba, fp)    
with open("{}\\base_results.pkl".format(output_folder), "wb") as fp:
    pickle.dump(base_results, fp)          
with open("{}\\base_roc.pkl".format(output_folder), "wb") as fp:
    pickle.dump(rocauc, fp)
with open("{}\\base_prc.pkl".format(output_folder), "wb") as fp:
    pickle.dump(prc, fp)     
with open("{}\\base_prc_thresh.pkl".format(output_folder), "wb") as fp:
    pickle.dump(prc_thresh, fp)   
        
        