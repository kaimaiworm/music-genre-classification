import pathlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, precision_score,  recall_score, f1_score, roc_auc_score, accuracy_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
# import packages for hyperparameters tuning
import optuna
from sklearn.model_selection import StratifiedKFold

from optuna.visualization.matplotlib import plot_optimization_history
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.model_selection import cross_validate
import pickle
from hyperparameter_stack import objective, stack_tuning
from stacking_functions import train_oof_predictions, model_selector, create_meta_dataset, stack_prediction
from sklearn.linear_model import LogisticRegression
import scikitplot as skplt
import time
import copy
from xgboost import plot_importance
import seaborn as sns
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
## Folders ##
################

data_input_folder    = "{}\output".format(path.__str__().replace("stacking", "dataset"))
base_input_folder    = "{}\output".format(path.__str__().replace("stacking", "base_models"))
                                            
output_folder        = "{}\\output".format(path)


################
# Load Dataset #
################

# Data for Rock Vs Other Classification
df_train = pd.read_excel("{}\\genres_bi.xlsx".format(data_input_folder), sheet_name = 0)
df_test = pd.read_excel("{}\\genres_bi.xlsx".format(data_input_folder), sheet_name = 1)

X_train = df_train.drop(["Unnamed: 0", "id", "artist name", "track name", "class"], axis = 1)
y_train = df_train["class"]

X_test = df_test.drop(["Unnamed: 0", "id", "artist name", "track name", "class"], axis = 1)
y_test = df_test["class"]



######################## 
#### Model Stacking ####
########################
##### Load needed model dictionary that includes optimized base_params
with open("{}\\models_dict_stack.pkl".format(base_input_folder), "rb") as fp:
        models_dict_stack = pickle.load(fp) 

##### OOF predictions
# init dicts for data and models
data_y = {}
trained_models = {} #stores models and oof predictions
data_x, data_y, trained_models = train_oof_predictions(X_train, y_train,  models_dict_stack)       
        
##### Model selection
# Set up a scoring dictionary to hold the model stack selector results
scores = {}
scores['Model'] = []
scores['F1'] = []
scores['Included'] = []

# Run the model stack selector for each model in trained_models
for model in trained_models:    
    meta_model = trained_models[model][0]
    label = model   
    resulting_models, best_acc = model_selector(data_x, data_y,  meta_model, trained_models, label)
    scores["Model"].append(model)
    scores["F1"].append(best_acc)
    scores["Included"].append(resulting_models)
    
# Transform scoreboard from dataframe to dictionary
best_models = {}
best_models = pd.DataFrame(scores)
best_models = best_models.sort_values("F1", ascending=False).reset_index(drop=True)

### Save best models
with open("{}\\best_stacks.pkl".format(output_folder), "wb") as fp:
    pickle.dump(best_models, fp)   
 
##### Load needed model dictionary that includes optimized base_params
with open("{}\\best_stacks.pkl".format(output_folder), "rb") as fp:
       best_models = pickle.load(fp)   
       
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
for idx in best_models.index:
    meta = best_models["Model"][idx] # idx: 0 = best models, 1 = second best etc.
    yhat_predics[meta] = []
    for model in best_models["Included"][idx]: 
        trained_models[model][0].fit(X_train_stack, y_train_stack) # fit base models to stack data (this is passed to function stack_prediction)
        yhat_predics[meta].append(trained_models[model][1]) # collect oof-predictions, obtained from before, as new variables
    
    # create the meta training data set using the oof predictions, call create_meta_dataset
    meta_X_train[meta] = create_meta_dataset(data_x, yhat_predics[meta])
    
    #create list of final base models
    final_models[meta] = []
    for model in best_models["Included"][idx]: # idx: 0 = best models, 1 = second best etc.
        final_models[meta].append(trained_models[model][0]) # append fitted base models to final_models dict
    
    #create the meta test data set using oof predictions, call stack_prediction
    meta_X_test[meta] = stack_prediction(X_test, final_models[meta])


##############################
#### Tuning Stacked Models ###
##############################

# init meta model and parameter dict
stack_params = {} 

models = ["logit", "knn", "svm", "lda", "qda", "dt", "rf", "gbt"]  # list of model names

stack_params = stack_tuning(meta_X_train, y_train_stack, models) #hyperparameter tuning

## Save best parameters
with open("{}\\stack_params.pkl".format(output_folder), "wb") as fp:
    pickle.dump(stack_params, fp)
    
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
gbt = XGBClassifier(objective="binary:logistic", booster = "gbtree", **stack_params["gbt"], eval_metric="auc", random_state = 123, n_jobs = -1)
rf = RandomForestClassifier(**stack_params["rf"], random_state = 123, n_jobs = -1)
logit = LogisticRegression(**stack_params["logit"], penalty = "elasticnet", solver = "saga", max_iter = 10000)        
dt = DecisionTreeClassifier(**stack_params["dt"], random_state = 123)
knn = KNeighborsClassifier(**stack_params["knn"], n_jobs = -1)
svm = SVC(**stack_params["svm"], probability = True)
qda = QuadraticDiscriminantAnalysis(**stack_params["qda"])
lda = LinearDiscriminantAnalysis(**stack_params["lda"], solver = "lsqr")

meta_models = {"logit": [logit],
                    "knn": [knn],
                    "svm": [svm],
                    "lda": [lda],
                    "qda" : [qda],
                    "dt": [dt],
                    "rf" : [rf], 
                    "gbt" : [gbt]
                    }


##########################    
### Optimal Thresholds ###
##########################

# Thresholds are chosen by maximizing F1-Score       

#init dicts
stack_thresholds = {}
proba_thresh = {}
prc_thresh = {}

for mod in meta_models.keys():
    # Use validation set to avoid data leakage on test set
    X1, X2, y1, y2 = train_test_split(meta_X_train[mod], y_train, test_size = 0.2, random_state=123)  
    
    meta_models[mod][0].fit(X1, y1) #fit model to training data
    proba_thresh[mod] = meta_models[mod][0].predict_proba(X2) # predict probabilities on validation data
    
    pre, rec, thresholds_prc = precision_recall_curve(y2, proba_thresh[mod][:, 1]) #calculate precision recall metrics
    prc_thresh[mod] = [pre, rec, thresholds_prc] #save metrics
    numer = (2 * prc_thresh[mod][0] * prc_thresh[mod][1]) #numerator of f1
    denom = (prc_thresh[mod][0]  + prc_thresh[mod][1] ) #denominator of f1
    fscore = np.divide(numer, denom, out=np.zeros_like(denom), where=(denom!=0)) #denominator of F1-Score can be zero, hence replace resulting nan with 0 in fscroe
    idx = np.argmax(fscore) #locate largest F1 Score
    stack_thresholds[mod] = prc_thresh[mod][2][idx] #save threshold that produces largest F1 Score
    
    
###### Calculate evaluation metrics for all models ######
# Calculate metrics
stack_preds = {}
stack_proba = {}
metrics_stack = []
rocauc = {}
prc = {}
    
for mod in meta_models.keys():
    meta_models[mod][0].fit(meta_X_train[mod], y_train_stack) #fit model to training data
    stack_proba[mod] = meta_models[mod][0].predict_proba(meta_X_test[mod])[:, 1] # predict probabilities
    stack_preds[mod] = copy.deepcopy(stack_proba[mod])
    
    stack_preds[mod][stack_preds[mod]>=stack_thresholds[mod]] = 1 #make predictions based on thresholds
    stack_preds[mod][stack_preds[mod]<stack_thresholds[mod]] = 0 
    
    #stack_preds[mod] = meta_models[mod][0].predict(meta_X_test[mod]) #not using optimized thresholds 
    
    #calculate metrics
    acc = np.round(accuracy_score(y_test, stack_preds[mod]), 4)
    precision = np.round(precision_score(y_test, stack_preds[mod], average= 'binary'), 4)
    recall = np.round(recall_score(y_test, stack_preds[mod], average= 'binary'), 4)
    f1 = np.round(f1_score(y_test, stack_preds[mod], average= 'binary'), 4)
    auc = np.round(roc_auc_score(y_test, stack_proba[mod]), 3)
    fpr, tpr, thresholds_roc = roc_curve(y_test, stack_proba[mod])
    pre, rec, thresholds_prc = precision_recall_curve(y_test, stack_proba[mod])
    
    rocauc[mod] = [fpr, tpr, thresholds_roc, auc]
    prc[mod] = [pre, rec, thresholds_prc]
    
    temp = [mod, acc, auc, precision, recall, f1]
    metrics_stack.append(temp)

stack_results = pd.DataFrame(metrics_stack, columns= ['Model', "Accuracy", "ROC-AUC", "Precision","Recall", "F1"])
stack_results.sort_values(by= ['F1'], ascending = False, inplace= True)
stack_results.reset_index(drop = True, inplace=True)

    
    
   

## Save models, predictions and results
with open("{}\\stack_models.pkl".format(output_folder), "wb") as fp:
    pickle.dump(meta_models, fp)
with open("{}\\stack_proba.pkl".format(output_folder), "wb") as fp:
    pickle.dump(stack_proba, fp)      
with open("{}\\stack_preds.pkl".format(output_folder), "wb") as fp:
    pickle.dump(stack_preds, fp)  
with open("{}\\stack_results.pkl".format(output_folder), "wb") as fp:
    pickle.dump(stack_results, fp)
with open("{}\\prc_thresh.pkl".format(output_folder), "wb") as fp:
    pickle.dump(prc_thresh, fp)   
with open("{}\\stack_roc.pkl".format(output_folder), "wb") as fp:
    pickle.dump(rocauc, fp)
with open("{}\\stack_prc.pkl".format(output_folder), "wb") as fp:
    pickle.dump(prc, fp)
with open("{}\\meta_X_train.pkl".format(output_folder), "wb") as fp:
    pickle.dump(meta_X_train, fp)
with open("{}\\meta_X_test.pkl".format(output_folder), "wb") as fp:
    pickle.dump(meta_X_test, fp)    
        
        