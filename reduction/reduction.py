import pathlib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
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
from stacking_functions import train_oof_predictions, create_meta_dataset, stack_prediction
from sklearn.utils import class_weight
from sklearn.model_selection import cross_validate
import pickle
from hyperparameter_reduction import objective, tuning
from sklearn.linear_model import LogisticRegression
import scikitplot as skplt
import time
from sklearn.preprocessing import LabelBinarizer
from xgboost import plot_importance
from sklearn.decomposition import PCA
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

data_input_folder   = "{}\output".format(path.__str__().replace("reduction", "dataset"))
stack_input_folder   = "{}\output".format(path.__str__().replace("reduction", "stacking"))

output_folder       = "{}\\output".format(path)

################
# Load Dataset #
################

# Data for Pop Vs Other Classification
df_train = pd.read_excel("{}\\genres_multi.xlsx".format(data_input_folder), sheet_name = 0)
df_test = pd.read_excel("{}\\genres_multi.xlsx".format(data_input_folder), sheet_name = 1)

X_train = df_train.drop(["Unnamed: 0", "id", "artist name", "track name", "class"], axis = 1)
y_train = df_train["class"]

X_test = df_test.drop(["Unnamed: 0", "id", "artist name", "track name", "class"], axis = 1)
y_test = df_test["class"]

###############################
### Dimensonality Reduction ###
###############################
n = 6  #number of components 

pca = PCA(n_components=n, random_state=123)
lda = LinearDiscriminantAnalysis(n_components = n)
reduc = {"pca": pca,
         "lda": lda}

X_train_reduc = {}
X_test_reduc = {}

for dim in reduc.keys(): 
    X_train_reduc[dim] = reduc[dim].fit(X_train, y_train).transform(X_train)
    X_test_reduc[dim] = reduc[dim].transform(X_test)


############################
###### Tune ML Models ######
############################

# Define list of models used for prediction
models = ["gbt", "knn", "lda", "qda", "dt"] 

# Init dict to store tuning parameters
best_params_base = {}
timer = {}

for dim in [reduc.keys()]:
    best_params_base[dim] = {}
    best_params_base[dim] = {}
    
    # Tune models for each lead-type
    tuner = objective(X_train_reduc[dim], y_train) #define objective 
    best_params_base[dim], timer[dim] = tuning(tuner, models) # store

## Save best parameters
with open("{}\\best_params_base_reduc.pkl".format(output_folder), "wb") as fp:
    pickle.dump(best_params_base, fp)
with open("{}\\timer_reduc.pkl".format(output_folder), "wb") as fp:
    pickle.dump(timer, fp)
    
with open("{}\\timer_reduc.pkl".format(output_folder), "rb") as fp:
    timer = pickle.load(fp)  
with open("{}\\best_params_base_reduc.pkl".format(output_folder), "rb") as fp:
    best_params_base = pickle.load(fp)      
#######################################
#### Init base models for stacking ####
#######################################

###### Initialize models
# init dictionary
models_dict_base = {}

# Needed later for model stacking, but easier to setup here
gbt_yhat, dt_yhat, lda_yhat, qda_yhat, knn_yhat = [], [], [], [], []

for dim in reduc.keys():
    models_dict_base[dim] = {}
    
    #initiate models and store in dict, use optuna parameters
    gbt = XGBClassifier(objective="multi:softprob", **best_params_base[dim]["gbt"], eval_metric="auc", random_state = 123, n_jobs = -1)
    dt = DecisionTreeClassifier(**best_params_base[dim]["dt"], random_state = 123)
    knn = KNeighborsClassifier(**best_params_base[dim]["knn"], n_jobs = -1)
    qda = QuadraticDiscriminantAnalysis(**best_params_base[dim]["qda"])
    lda = LinearDiscriminantAnalysis(**best_params_base[dim]["lda"], solver = "lsqr")
    
    models_dict_base[dim] = {"knn": [knn, knn_yhat],
                        "lda": [lda, lda_yhat],
                        "qda" : [qda, qda_yhat],
                        "dt": [dt, dt_yhat],
                        "gbt" : [gbt, gbt_yhat]
                        }    

models_dict_stack = copy.deepcopy(models_dict_base) #unfitted copy for latter

####################################
### Final prediction on test set ###
####################################

###### Calculate evaluation metrics for all models ######
# init dicts for storage
base_preds = {}
base_proba = {}
metrics_base = {}
base_results = {}

# predict on data 

for dim in reduc.keys():
    base_preds[dim] = {}
    base_proba[dim] = {}
    metrics_base[dim] = []
    
    for mod in models_dict_base[dim].keys():
        
        models_dict_base[dim][mod][0].fit(X_train_reduc[dim], y_train) #fit model to training data
        
        base_proba[dim][mod] = models_dict_base[dim][mod][0].predict_proba(X_test_reduc[dim]) # predict probabilities
        
        base_preds[dim][mod] = models_dict_base[dim][mod][0].predict(X_test_reduc[dim]) #predict classes
        
        #calculate metrics
        acc = np.round(accuracy_score(y_test, base_preds[dim][mod]), 4)
        precision = np.round(precision_score(y_test, base_preds[dim][mod], average= "weighted"), 4)
        recall = np.round(recall_score(y_test, base_preds[dim][mod], average= "weighted"), 4)
        f1 = np.round(f1_score(y_test, base_preds[dim][mod], average= "weighted"), 4)
        auc = np.round(roc_auc_score(y_test, base_proba[dim][mod], multi_class= "ovr"), 3)
        
        temp = [mod, timer[dim][mod], acc, auc, precision, recall, f1] #gather metrics
        metrics_base[dim].append(temp)
    
    base_results[dim] = pd.DataFrame(metrics_base[dim], columns= ["Model", "Time", "Accuracy", "ROC-AUC", "Precision","Recall", "F1"])
    base_results[dim].sort_values(by= ['F1'], ascending = False, inplace= True)
    base_results[dim].reset_index(drop = True, inplace=True)


         

## Save models, predictions and results1

with open("{}\\base_models_reduc.pkl".format(output_folder), "wb") as fp:
    pickle.dump(models_dict_base, fp)
with open("{}\\base_proba_reduc.pkl".format(output_folder), "wb") as fp:
    pickle.dump(base_proba, fp)      
with open("{}\\base_preds_reduc.pkl".format(output_folder), "wb") as fp:
    pickle.dump(base_preds, fp)  
with open("{}\\base_results_reduc.pkl".format(output_folder), "wb") as fp:
    pickle.dump(base_results, fp)  

