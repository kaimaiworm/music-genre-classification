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
from stacking_functions import train_oof_predictions, create_meta_dataset, stack_prediction
from sklearn.utils import class_weight
from sklearn.model_selection import cross_validate
import pickle
from hyperparameter_multi import objective, tuning, stack_tuning
from sklearn.linear_model import LogisticRegression
import scikitplot as skplt
import time
from sklearn.preprocessing import LabelBinarizer
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

data_input_folder   = "{}\output".format(path.__str__().replace("multiclass", "dataset"))
stack_input_folder   = "{}\output".format(path.__str__().replace("multiclass", "stacking"))

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


############################
###### Tune ML Models ######
############################

# Define list of models used for prediction
models = ["gbt", "knn", "lda", "qda", "dt"] 

# Init dict to store tuning parameters
best_params_base = {}

# Tune models for each lead-type
tuner = objective(X_train, y_train) #define objective 
best_params_base, timer = tuning(tuner, models) # store

## Save best parameters
with open("{}\\best_params_base_multi.pkl".format(output_folder), "wb") as fp:
    pickle.dump(best_params_base, fp)

#######################################
#### Init base models for stacking ####
#######################################

###### Initialize models
# init dictionary
models_dict_base = {}

# Needed later for model stacking, but easier to setup here
gbt_yhat, dt_yhat, lda_yhat, qda_yhat, knn_yhat = [], [], [], [], []

#initiate models and store in dict, use optuna parameters
gbt = XGBClassifier(objective="multi:softprob", **best_params_base["gbt"], eval_metric="auc", random_state = 123, n_jobs = -1)
dt = DecisionTreeClassifier(**best_params_base["dt"], random_state = 123)
knn = KNeighborsClassifier(**best_params_base["knn"], n_jobs = -1)
qda = QuadraticDiscriminantAnalysis(**best_params_base["qda"])
lda = LinearDiscriminantAnalysis(**best_params_base["lda"], solver = "lsqr")

models_dict_base = {"knn": [knn, knn_yhat],
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
metrics_base = []
rocauc = {}
prc = {}

# predict on data 
for mod in models_dict_base.keys():
    models_dict_base[mod][0].fit(X_train, y_train) #fit model to training data
    
    base_proba[mod] = models_dict_base[mod][0].predict_proba(X_test) # predict probabilities
    
    base_preds[mod] = models_dict_base[mod][0].predict(X_test) #predict classes
    
    #calculate metrics
    acc = np.round(accuracy_score(y_test, base_preds[mod]), 4)
    precision = np.round(precision_score(y_test, base_preds[mod], average= "weighted"), 4)
    recall = np.round(recall_score(y_test, base_preds[mod], average= "weighted"), 4)
    f1 = np.round(f1_score(y_test, base_preds[mod], average= "weighted"), 4)
    auc = np.round(roc_auc_score(y_test, base_proba[mod], multi_class= "ovr"), 3)
    
    temp = [mod, timer[mod], acc, auc, precision, recall, f1] #gather metrics
    metrics_base.append(temp)

base_results = pd.DataFrame(metrics_base, columns= ["Model", "Time", "Accuracy", "ROC-AUC", "Precision","Recall", "F1"])
base_results.sort_values(by= ['F1'], ascending = False, inplace= True)
base_results.reset_index(drop = True, inplace=True)

### ROC and PRC curves by using One vs Rest method

label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_test) 

for mod in models_dict_base.keys():
    
    rocauc[mod] = {}
    prc[mod] = {}
    
    for class_id in label_binarizer.classes_:
        
        fpr, tpr, thresholds_roc = roc_curve(y_onehot_test[:, class_id], base_proba[mod][:, class_id])
        pre, rec, thresholds_prc = precision_recall_curve(y_onehot_test[:, class_id], base_proba[mod][:, class_id])
    
        rocauc[mod] = [fpr, tpr, thresholds_roc, auc]
        prc[mod] = [pre, rec, thresholds_prc]
    
######################## 
#### Model Stacking ####
########################

##### Load best model stacks from binary case
with open("{}\\best_stacks.pkl".format(stack_input_folder), "rb") as fp:
        best_stacks = pickle.load(fp) 

best_stacks = best_stacks.loc[best_stacks["Model"].isin(["qda", "gbt"])] #only use two best models

##### OOF predictions
# init dicts for data and models
data_y = {}
trained_models = {} #stores models and oof predictions
data_x, data_y, trained_models = train_oof_predictions(X_train, y_train,  models_dict_stack)           


############################
### Create Meta Datasets ###
############################
      
##### Init new training data frames for stacking 
X_train_stack = pd.DataFrame(data_x, columns = X_train.columns)
y_train_stack = pd.Series(data_y)

# Init dict for final oof predics
yhat_predics = {}
meta_X_train = {}
meta_X_test = {}
final_models = {}

### Create meta training and test data
#loop over all stacks
for idx in best_stacks.index:
    meta = best_stacks["Model"][idx] # idx: 0 = best models, 1 = second best 
    yhat_predics[meta] = []
    for model in best_stacks["Included"][idx]: 
        trained_models[model][0].fit(X_train_stack, y_train_stack) # fit base models to stack data (this is passed to function stack_prediction)
        yhat_predics[meta].append(trained_models[model][1]) # collect oof-predictions, obtained from before, as new variables
    
    # create the meta training data set using the oof predictions, call create_meta_dataset
    meta_X_train[meta] = create_meta_dataset(data_x, yhat_predics[meta])
    
    #create list of final base models
    final_models[meta] = []
    for model in best_stacks["Included"][idx]: # idx: 0 = best models, 1 = second best etc.
        final_models[meta].append(trained_models[model][0]) # append fitted base models to final_models dict
    
    #create the meta test data set using oof predictions, call stack_prediction
    meta_X_test[meta] = stack_prediction(X_test, final_models[meta])

    

##############################
#### Tuning Stacked Models ###
##############################

# init meta model and parameter dict
stack_params = {} 

models = ["gbt", "qda"] 

stack_params = stack_tuning(meta_X_train, y_train_stack, models) #hyperparameter tuning

## Save best parameters
with open("{}\\stack_params.pkl".format(output_folder), "wb") as fp:
    pickle.dump(stack_params, fp)

##### Load saved stack
with open("{}\\stack_params.pkl".format(output_folder), "rb") as fp:
        stack_params = pickle.load(fp)  
        
################################
### Stacked Model Prediction ###
################################


#### Calculate metrics for test period
# init dictionary 
meta_models = {}

#init models with best stack params
# initiate models and store in dict, use parameters from optuna
gbt = XGBClassifier(objective="multi:softprob", booster = "gbtree", **stack_params["gbt"], eval_metric="auc", random_state = 123, n_jobs = -1)
qda = QuadraticDiscriminantAnalysis(**stack_params["qda"])

meta_models = {"qda" : [qda],
               "gbt" : [gbt]
               }

###### Calculate evaluation metrics for all models ######
# Calculate metrics
stack_preds = {}
stack_proba = {}
metrics_stack = []
rocauc_stack = {}
prc_stack = {}
    
for mod in meta_models.keys():
    meta_models[mod][0].fit(meta_X_train[mod], y_train_stack) #fit model to training data
    
    stack_proba[mod] = meta_models[mod][0].predict_proba(meta_X_test[mod]) # predict probabilities
    
    stack_preds[mod] = meta_models[mod][0].predict(meta_X_test[mod]) #predict classes
    
    #calculate metrics
    acc = np.round(accuracy_score(y_test, stack_preds[mod]), 4)
    precision = np.round(precision_score(y_test, stack_preds[mod], average= "weighted"), 4)
    recall = np.round(recall_score(y_test, stack_preds[mod], average= "weighted"), 4)
    f1 = np.round(f1_score(y_test, stack_preds[mod], average= "weighted"), 4)
    auc = np.round(roc_auc_score(y_test, stack_proba[mod], multi_class = "ovo"), 3)

    temp = [mod, acc, auc, precision, recall, f1]
    metrics_stack.append(temp)

stack_results = pd.DataFrame(metrics_stack, columns= ['Model', "Accuracy", "ROC-AUC", "Precision","Recall", "F1"])
stack_results.sort_values(by= ['F1'], ascending = False, inplace= True)
stack_results.reset_index(drop = True, inplace=True)

### ROC and PRC curves by using One vs Rest method

for mod in meta_models.keys():
    rocauc_stack[mod] = {}
    prc_stack[mod] = {}
    
    for class_id in label_binarizer.classes_:
        
        fpr, tpr, thresholds_roc = roc_curve(y_onehot_test[:, class_id], stack_proba[mod][:, class_id])
        pre, rec, thresholds_prc = precision_recall_curve(y_onehot_test[:, class_id], stack_proba[mod][:, class_id])
    
        rocauc_stack[mod] = [fpr, tpr, thresholds_roc, auc]
        prc_stack[mod] = [pre, rec, thresholds_prc]
         

## Save models, predictions and results

#Base Models

with open("{}\\base_models_multi.pkl".format(output_folder), "wb") as fp:
    pickle.dump(models_dict_base, fp)
with open("{}\\base_proba_multi.pkl".format(output_folder), "wb") as fp:
    pickle.dump(base_proba, fp)      
with open("{}\\base_preds_multi.pkl".format(output_folder), "wb") as fp:
    pickle.dump(base_preds, fp)  
with open("{}\\base_results_multi.pkl".format(output_folder), "wb") as fp:
    pickle.dump(base_results, fp)  
with open("{}\\base_roc_multi.pkl".format(output_folder), "wb") as fp:
    pickle.dump(rocauc, fp)
with open("{}\\base_prc_multi.pkl".format(output_folder), "wb") as fp:
    pickle.dump(prc, fp)
    
# Stacked models    
with open("{}\\stack_models_multi.pkl".format(output_folder), "wb") as fp:
    pickle.dump(meta_models, fp)
with open("{}\\stack_proba_multi.pkl".format(output_folder), "wb") as fp:
    pickle.dump(stack_proba, fp)      
with open("{}\\stack_preds_multi.pkl".format(output_folder), "wb") as fp:
    pickle.dump(stack_preds, fp)  
with open("{}\\stack_results_multi.pkl".format(output_folder), "wb") as fp:
    pickle.dump(stack_results, fp)  
with open("{}\\stack_roc_multi.pkl".format(output_folder), "wb") as fp:
    pickle.dump(rocauc_stack, fp)
with open("{}\\stack_prc_multi.pkl".format(output_folder), "wb") as fp:
    pickle.dump(prc_stack, fp)