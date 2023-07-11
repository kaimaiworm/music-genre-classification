import pathlib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score,  recall_score, f1_score, roc_auc_score, accuracy_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
# import packages for hyperparameters tuning
import optuna
from sklearn.model_selection import StratifiedKFold

from optuna.visualization.matplotlib import plot_optimization_history

from sklearn.utils import class_weight
from sklearn.model_selection import cross_validate

from hyperparameter import objective
from sklearn.linear_model import LogisticRegression
import scikitplot as skplt
import time
from xgboost import plot_importance
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.io as pio
pio.renderers.default='browser'

import warnings
warnings.filterwarnings('ignore')

# current working directory
path = pathlib.Path().absolute()

complete_start = time.time()
################
### Folders ####
################

input_folder         = "{}\\input".format(path)
output_folder        = "{}\\output".format(path)

################
# Load Dataset #
################

# Data for Rock Vs Other Classification
df_train_bi = pd.read_excel("{}\\genres_bi.xlsx".format(input_folder), sheet_name = 0)
df_test_bi = pd.read_excel("{}\\genres_bi.xlsx".format(input_folder), sheet_name = 1)

x_train = df_train_bi.drop(["id", "artist name", "track name", "class"], axis = 1)
y_train = df_train_bi["class"]

x_test = df_test_bi.drop(["id", "artist name", "track name", "class"], axis = 1)
y_test = df_test_bi["class"]


#### Initialize objective function for hyperparameter tuning

tuning_objective = objective(x_train, y_train, x_test, y_test)

### XGB

xgb_start = time.time()

np.random.seed(42)
study_xgb = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
)
study_xgb.optimize(tuning_objective.xgb, n_trials=100, timeout=600)

xgb_time = time.time() - xgb_start


# Save best parameters
xgb_params = study_xgb.best_params

# instantiate the classifier 
xgb = XGBClassifier(objective="binary:logistic", **xgb_params, eval_metric="auc", early_stopping_rounds=10, random_state = 123, seed = 0)


# fit the classifier to the training data
evaluation = [( x_train, y_train), ( x_test, y_test)]
xgb.fit(x_train, y_train, eval_set=evaluation, verbose=False)

# make predictions with test data
xgb_pred = xgb.predict(x_test)

xgb_acc = f1_score(y_test, xgb_pred, average="binary")

##########################

### Logit

logit_start = time.time()
np.random.seed(42)
study_logit = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
)
study_logit.optimize(tuning_objective.logit, n_trials=100, timeout=600)

logit_time = time.time() - logit_start

logit_params = study_logit.best_params

# instantiate the classifier 
logit = LogisticRegression(solver="saga", max_iter = 10000, **logit_params, random_state=123)

# fit the classifier to the training data
logit.fit(x_train, y_train)

# make predictions with test data
logit_pred = logit.predict(x_test)

logit_acc = f1_score(y_test, logit_pred)


#########################

### Decision Tree

dt_start = time.time()
np.random.seed(42)
study_dt = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
)
study_dt.optimize(tuning_objective.dt, n_trials=100, timeout=600)

dt_time = time.time() - dt_start

dt_params = study_dt.best_params

# instantiate the classifier 
dt = DecisionTreeClassifier(**dt_params, random_state = 123)

# fit the classifier to the training data
dt.fit(x_train, y_train)

# make predictions with test data
dt_pred = dt.predict(x_test)

dt_acc = f1_score(y_test, dt_pred)


##########################

### Random Forest

rf_start = time.time()
np.random.seed(42)
study_rf = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
)
study_rf.optimize(tuning_objective.rf, n_trials=100, timeout=600)

rf_time = time.time() - rf_start

rf_params = study_rf.best_params

# instantiate the classifier 
rf = RandomForestClassifier(**rf_params, random_state = 123)

# fit the classifier to the training data
rf.fit(x_train, y_train)

# make predictions with test data
rf_pred = rf.predict(x_test)

rf_acc = f1_score(y_test, rf_pred)

##########################

### Gradient Boosting

gb_start = time.time()
np.random.seed(42)
study_gb = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
)
study_gb.optimize(tuning_objective.gb, n_trials=100, timeout=600)

gb_time = time.time() - gb_start

gb_params = study_gb.best_params

# instantiate the classifier 
gb = GradientBoostingClassifier(**gb_params, random_state = 123)

# fit the classifier to the training data
gb.fit(x_train, y_train)

# make predictions with test data
gb_pred = gb.predict(x_test)

gb_acc = f1_score(y_test, gb_pred)

##########################

"""


rf_params = {'max_depth': 10,
 'criterion': 'entropy',
 'max_features': 'sqrt',
 'min_samples_leaf': 3,
 'min_samples_split': 3,
 'n_estimators': 50}

xgb_params = {'max_depth': 4,
 'gamma': 6.441084559421174,
 'alpha': 27,
 'lambda': 0.855290174859911,
 'colsample_bytree': 0.8451028736731392,
 'min_child_weight': 9,
 'learning_rate': 0.3903181760015277,
 'n_estimators': 800}

gb_params = {'max_depth': 10,
 'learning_rate': 0.08160500477509855,
 'criterion': 'friedman_mse',
 'loss': 'exponential',
 'max_features': 'sqrt',
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'n_estimators': 250,
 'min_weight_fraction_leaf': 0.11428173080118341}

logit_params = {'penalty': 'elasticnet',
 'C': 0.010271591143171516,
 'l1_ratio': 0.9829404117868408}

dt_params = {'max_depth': 10,
 'criterion': 'gini',
 'max_features': None,
 'min_samples_leaf': 2,
 'min_samples_split': 3,
 'splitter': 'best'}

rf_time = 1
gb_time = 12
xgb_time = 3
dt_time = 4
logit_time = 4
"""

##############################

####### Model selection for stacking #######

rf_yhat, gb_yhat, xgb_yhat, logit_yhat, dt_yhat = [], [], [], [], []

gb = GradientBoostingClassifier(**gb_params, random_state = 123)
rf = RandomForestClassifier(**rf_params, random_state = 123)
logit = LogisticRegression(solver="saga", max_iter = 10000, **logit_params, random_state=123)
xgb = XGBClassifier(objective="binary:logistic", **xgb_params, eval_metric="auc", random_state = 123, seed = 0)
dt = DecisionTreeClassifier(**dt_params, random_state = 123)

    
models_dict = {'RF' : [rf, rf_yhat], 
                'GB' : [gb, gb_yhat],  
                'XGB' : [xgb, xgb_yhat],
                "Logit": [logit, logit_yhat],
                "DT": [dt, dt_yhat]}



def train_oof_predictions(x, y, models, verbose=True):
    '''Function to perform Out-Of-Fold predictions on train data
    returns re-ordered predictors x, re-ordered target y, and model dictionary with filled predictors
    Parameters:
    x: training predictors
    y: training targets
    models: dictionary of models in form of model name : [instantiated model, predictors list]
    verbose: if True, prints status update as the function works
    '''
    
    # instantiate a KFold with 10 splits
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # prepare lists to hold the re-ordered x and y values
    data_x, data_y  = [], []
    
    x_arr = np.array(x)
    y_arr = np.array(y)
    # run the following block for each of the kfold splits

    for train_ix, test_ix in kfold.split(x_arr, y_arr):
    
        if verbose: print("\nStarting a new fold\n")
    
        if verbose: print("Creating splits")
        #create this fold's training and test sets
        train_X, test_X = x_arr[train_ix], x_arr[test_ix] 
        train_y, test_y = y_arr[train_ix], y_arr[test_ix]
    
        if verbose: print("Adding x and y to lists\n")
        # add the data that is used in this fold to the re-ordered lists
        data_x.extend(test_X)
        data_y.extend(test_y)
    
        # run each model on this kfold and add the predictors to the model's running predictors list
        for item in models:
            
            label = item # get label for reporting purposes
            model = models[item][0] # get the model to use on the kfold
                            
            # fit and make predictions 
            if verbose: print("Running",label,"on this fold")
            if item == "XGB":
                # fit the classifier to the training data, use evaluation set
                model = XGBClassifier(objective="binary:logistic", **xgb_params, eval_metric="auc", early_stopping_rounds=10, random_state = 123, seed = 0)
                evaluation = [( train_X, train_y), ( test_X, test_y)]
                model.fit(train_X, train_y, eval_set=evaluation, verbose=False)
                predictions = model.predict(test_X) # fit on the out-of-fold set
                models[item][1].extend(predictions) # add predictions to the model's running predictors list
            else:    
                model.fit(train_X, train_y) # fit to the train set for the kfold
                predictions = model.predict(test_X) # fit on the out-of-fold set
                models[item][1].extend(predictions) # add predictions to the model's running predictors list
    
    return data_x, data_y, models

data_x, data_y, trained_models = train_oof_predictions(x_train, y_train, models_dict)

def model_selector(X, y, meta_model, models_dict, model_label, verbose=True):
    
    print("\n\nRunning model selector for ", model_label, "as meta-model")
    included_models = []
     
    while True:
        changed=False
        
        # forward step
        
        if verbose: print("\nNEW ROUND - Setting up score charts")
        excluded_models = list(set(models_dict.keys())-set(included_models)) # make a list of the current excluded_models
        if verbose: print("Included models: {}".format(included_models))
        if verbose: print("Exluded models: {}".format(excluded_models))
        new_acc = pd.Series(index=excluded_models) # make a series where the index is the current excluded_models
        
        current_meta_x = np.array(X)
        
        if len(included_models) > 0:
            for included in included_models:
                included = np.array(models_dict[included][1]).reshape((len(models_dict[included][1]), 1))
                current_meta_x = np.hstack((current_meta_x, included))# score the current model
        scores = cross_validate(meta_model, current_meta_x, y, cv=5, n_jobs=-1, scoring=('f1'))
        starting_acc = round(scores['test_score'].mean(),3)
        if verbose: print("Starting accuracy: {}\n".format(starting_acc))
        
       
        for excluded in excluded_models:  # for each item in the excluded_models list:
            
            new_yhat = np.array(models_dict[excluded][1]).reshape(-1, 1) # get the current item's predictions
            meta_x = np.hstack((current_meta_x, new_yhat)) # add the predictions to the meta set
            
            # score the current item
            scores = cross_validate(meta_model, meta_x, y, cv=5, n_jobs=-1, scoring=('f1'))
            acc = round(scores['test_score'].mean(),3)
            if verbose: print("{} score: {}".format(excluded, acc))
            
            new_acc[excluded] = acc # append the accuracy to the series field
        
        best_acc = new_acc.max() # evaluate best accuracy of the excluded_models in this round
        if verbose: print("Best acc: {}\n".format(best_acc))
        
        if best_acc > starting_acc:  # if the best acc is better than the initial acc
            best_feature = new_acc.idxmax()  # define this as the new best feature
            included_models.append(str(best_feature)) # append this model name to the included list
            changed=True # flag that we changed it
            if verbose: print('Add  {} with accuracy {}\n'.format(best_feature, best_acc))
        else: changed = False
        
        if not changed:
            break
            
    print(model_label, "model optimized")
    print('resulting models:', included_models)
    print('Accuracy:', starting_acc)
    
    return included_models, starting_acc


# Set up a scoring dictionary to hold the model stack selector results
scores = {}
scores['Model'] = []
scores['Accuracy'] = []
scores['Included'] = []

# Run the model stack selector for each model in our trained_models
for model in trained_models:    
    meta_model = trained_models[model][0]
    label = model   
    resulting_models, best_acc = model_selector(data_x, data_y,  meta_model, trained_models, label, verbose=True)
    scores['Model'].append(model)
    scores['Accuracy'].append(best_acc)
    scores['Included'].append(resulting_models)



# Look at the scores of our model combinations 
best_model = pd.DataFrame(scores)
best_model = best_model.sort_values("Accuracy", ascending=False).reset_index(drop=True)


###### Model Stacking #######

# Check our meta model on the original train/test set only
# Instantiate the chosen meta model
#meta_model = trained_models[best_model["Model"][0]][0] # 0 for best model, 1 for second best etc.

meta_model = XGBClassifier(objective="binary:logistic", **xgb_params, eval_metric="auc", early_stopping_rounds=10, random_state = 123, seed = 0)

evaluation = [( x_train, y_train), ( x_test, y_test)]
meta_model.fit(x_train, y_train, eval_set=evaluation, verbose=False)
predictions = meta_model.predict(x_test)
acc_meta = f1_score(predictions, y_test, average = "binary")
print("Accuracy before stacking: ", acc_meta)

#Create list of predictions

stack_start = time.time()

yhat_predics = []

for model in best_model["Included"][0]: # 0 best best models, 1 for second best etc.
    print("Fitting model: {}".format(model))
    trained_models[model][0].fit(x_train, y_train)
    yhat_predics.append(trained_models[model][1])

def create_meta_dataset(data_x, items):
    '''Function that takes in a data set and list of predictions, and forges into one dataset
    parameters:
    data_x - original data set
    items - list of predictions
    returns: stacked data set
    '''
    
    meta_x = data_x
    
    for z in items:
        z = np.array(z).reshape((len(z), 1))
        meta_x = np.hstack((meta_x, z))
        
    return meta_x


# create the meta data set using the oof predictions
meta_X_train = create_meta_dataset(data_x, yhat_predics)

#create list of final base models
final_models = []
for model in best_model["Included"][0]: # 0 for best models, 1 for second best etc.
    final_models.append(trained_models[model][0])


def stack_prediction(x_test, final_models): 
    '''takes in a test set and a list of fitted models.
    Fits each model in the list on the test set and stores it in a predictions list. Then sends the test set and the predictions to the create_meta_dataset to be combined
    Returns: combined meta test set
    Parameters:
    X_test - testing dataset
    final_models - list of fitted models
    '''
    predictions = []
    
    for model in final_models:
        preds = model.predict(x_test).reshape(-1,1)
        predictions.append(preds)
    
    meta_X = create_meta_dataset(x_test, predictions)
        
    return meta_X


meta_X_test = stack_prediction(x_test, final_models)



# fit the meta model to the Train meta dataset
# There is no data leakage in the meta dataset since we did all of our predictions out-of-sample!
evaluation = [( meta_X_train, data_y), ( meta_X_test, y_test)]
meta_model.fit(meta_X_train, data_y, eval_set=evaluation, verbose=False) # predict on the meta test set
predictions = meta_model.predict(meta_X_test)
acc_stack = f1_score(predictions, y_test, average = "binary")
print("Accuracy of stacked model: ", acc_stack)


###Hyperparameter tuning on stacked model
### init objective function

stack_objective = objective(meta_X_train, data_y, meta_X_test, y_test)


np.random.seed(42)
study_stack = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
)
study_stack.optimize(stack_objective.xgb, n_trials=100, timeout=600)

stack_params = study_stack.best_params

stack_time = time.time() - stack_start 

# instantiate the classifier 
stack = XGBClassifier(objective="binary:logistic", **xgb_params, eval_metric="auc", early_stopping_rounds=10, random_state = 123, seed = 0)

# fit the classifier to the training data
evaluation = [( meta_X_train, data_y), ( meta_X_test, y_test)]
stack.fit(meta_X_train, data_y, eval_set=evaluation, verbose=False)

# make predictions with test data
stack_pred = stack.predict(meta_X_test)

stack_acc = f1_score(y_test, stack_pred, average = "binary")

print(acc_meta)
print(stack_acc)


###### Calculate evaluation metrics for all models ######

# Gather all models 
all_models = {}

for key in trained_models.keys():
    all_models[key] = trained_models[key][0]
all_models["Stack"] = stack
all_models["XGB"] = XGBClassifier(objective="binary:logistic", **xgb_params, eval_metric="auc", early_stopping_rounds=10, random_state = 123, seed = 0)

# Calculate metrics
metrics = []
rocauc = {}
prc = {}
threshold = 0.248137 

for key in all_models.keys():
    
    temp = []
    model = all_models[key]
    
    if key == "XGB":
        # fit the classifier to the training data, use evaluation set
        evaluation = [( x_train, y_train), ( x_test, y_test)]
        model.fit(x_train, y_train, eval_set=evaluation, verbose=False)
        y_proba = model.predict_proba(x_test)[:, 1]
        y_pred = np.copy(y_proba)
        y_pred[y_pred>=threshold] = 1
        y_pred[y_pred<threshold] = 0

        #y_pred = model.predict(x_test)
        
        
    elif key == "Stack":
        # fit the classifier to the meta data, use evaluation
        evaluation = [( meta_X_train, data_y), ( meta_X_test, y_test)]
        model.fit(meta_X_train, data_y, eval_set=evaluation, verbose=False)
        y_proba = model.predict_proba(meta_X_test)[:, 1]
        y_pred = np.copy(y_proba)
        y_pred[y_pred>=threshold] = 1
        y_pred[y_pred<threshold] = 0
        #y_pred = model.predict(meta_X_test)
    
    else:
        # fit the classifier to the training data
        model.fit(x_train, y_train)
        y_proba = model.predict_proba(x_test)[:, 1]
        y_pred = np.copy(y_proba)
        y_pred[y_pred>=threshold] = 1
        y_pred[y_pred<threshold] = 0
        #y_pred = model.predict(x_test)
        
    
    test_acc = np.round(accuracy_score(y_test, y_pred), 4)
    precision = np.round(precision_score(y_test, y_pred, average= 'binary'), 4)
    recall = np.round(recall_score(y_test, y_pred, average= 'binary'), 4)
    f1 = np.round(f1_score(y_test, y_pred, average= 'binary'), 4)
    auc = np.round(roc_auc_score(y_test, y_proba), 3)
    fpr1, tpr1, thresholds1 = roc_curve(y_test, y_proba)
    pre, rec, thresholds2 = precision_recall_curve(y_test, y_proba)
    
    rocauc[key] = [fpr1, tpr1, thresholds1, auc]
    prc[key] = [pre, rec, thresholds2]
    
    temp = [key, test_acc, precision, recall, f1]
    metrics.append(temp)
    
timer = [np.round(rf_time, 4), np.round(gb_time, 4), np.round(xgb_time, 4), np.round(logit_time, 4), np.round(dt_time, 4), np.round(stack_time, 4)]
    
df_results = pd.DataFrame(metrics, columns= ['Model', "Accuracy",'Precision','Recall', 'F1'])
df_results.insert(1, "Time", timer, True)
df_results.sort_values(by= ['F1'], inplace= True)
df_results.reset_index(drop = True, inplace=True)

"""
 # calculate the g-mean for each threshold
gmeans = np.sqrt(rocauc["Stack"][1] * (1-rocauc["Stack"][0]))
# locate the index of the largest g-mean
ix = np.argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (rocauc["Stack"][2][ix], gmeans[ix]))

# convert to f score
fscore = (2 * prc["Stack"][0] * prc["Stack"][1] ) / (prc["Stack"][0]  + prc["Stack"][1] )
# locate the index of the largest f score
ix = np.argmax(fscore)
print('Best Threshold=%f, F-Score=%.3f' % (prc["Stack"][2][ix], fscore[ix]))

"""

######## Create ROC curve for stacked model #########
y_probas = stack.predict_proba(meta_X_test)

plt.rcParams["figure.figsize"] = (15, 10)
roc = skplt.metrics.plot_roc(y_test, y_probas, plot_micro = False, plot_macro = False)
roc.tick_params(axis='x', labelsize=18)
roc.tick_params(axis='y', labelsize=18)
plt.ylabel('True Positive Rate', fontsize=20)
plt.xlabel('False Positive Rate', fontsize=20)
plt.legend(loc=4, prop={'size': 18})
plt.title(None)
plt.savefig("stacked roc.png", dpi = 800, bbox_inches='tight')

plt.clf()

######## ROC Curve for all models ########
plt.rcParams["figure.figsize"] = (15, 10)

for key in all_models.keys():
    plt.plot(rocauc[key][0], rocauc[key][1], linewidth = 2, label="{}, AUC={}".format(key, rocauc[key][3]))

plt.plot([0, 1], [0, 1], color = 'black', linewidth = 2, linestyle = "dotted")
plt.ylabel('True Positive Rate', fontsize=20)
plt.xlabel('False Positive Rate', fontsize=20)
plt.legend(loc=4, prop={'size': 18})
plt.title(None)
plt.savefig("all roc.png", dpi = 800, bbox_inches='tight')
plt.clf()

######## Hyperparemter tuning plot for XGB Model #########

plt.rcParams["figure.figsize"] = (15, 10)
plt.rc('axes', titlesize=50) 
opt_plot = optuna.visualization.matplotlib.plot_optimization_history(study_xgb)

plt.savefig("tuning.png", dpi = 800, bbox_inches='tight')
plt.clf()

######### XGBoost Feature importance parameter plot #########
# Get the booster from the xgbmodel
booster = xgb.get_booster()

# Get the importance dictionary (by gain) from the booster
importance = booster.get_score(importance_type="gain")

# make your changes
gain = []
feature = []
for key in importance.keys():
    gain.append(round(importance[key],2))
    feature.append(key)
    
imp = pd.DataFrame({"Information Gain": gain, "Feature": feature} )    
imp.sort_values("Information Gain", ascending = False, inplace = True)
imp.reset_index(drop = True, inplace=True)

# Scatter plot 
trace = go.Scatter(
    y = imp["Information Gain"].values[0:10], # Only first ten features 
    x = imp["Feature"].values[0:10],
    mode='markers+text',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = imp['Information Gain'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = imp['Information Gain'].values[0:10]
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= None,
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
fig.update_layout(
    font=dict(size=25)
    )

fig.update_traces(textposition='top center')
pio.write_image(fig, "xgb importance.png",scale=6, width=1500, height=1000)



############### Stack XGB Feature Importance #####################
# Get the booster from the xgbmodel
booster = stack.get_booster()

# Get the importance dictionary (by gain) from the booster
importance = booster.get_score(importance_type="gain")

# make your changes
gain = []
feature = []
for key in importance.keys():
    gain.append(round(importance[key],2))
    feature.append(key)    
    
imp = pd.DataFrame({"Information Gain": gain, "Feature": feature} )    
imp.sort_values("Information Gain", ascending = False, inplace = True)
imp.reset_index(drop = True, inplace=True)

feature_names = list(x_train.columns)
for x in ["GB_pred", "XGB_pred"]: feature_names.append(x)   

features_select = [feature_names[i] for i in (27,28,1,7,6,11,4,2)]
# Scatter plot 
trace = go.Scatter(
    y = imp["Information Gain"].values[0:10], # Only first ten features 
    x = features_select,
    mode='markers+text',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = imp['Information Gain'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = imp['Information Gain'].values[0:10]
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= None,
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Information Gain',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
fig.update_layout(
    font=dict(size=25)
    )

fig.update_traces(textposition='top center')
pio.write_image(fig, "stack importance.png",scale=6, width=1500, height=1000)

"""
########## Stack Feature Importance plot #########
feature_names = list(x_train.columns)
for x in ["GB_pred", "RF_pred", "XGB_pred"]: feature_names.append(x)
stack_gain = np.round(stack.feature_importances_, 2)
stack_imp = pd.DataFrame({"Information Gain": stack_gain, "Feature": feature_names} )    
stack_imp.sort_values("Information Gain", ascending = False, inplace = True)
stack_imp.reset_index(drop = True, inplace=True)

# Scatter plot 
trace = go.Scatter(
    y = stack_imp["Information Gain"].values[0:10], # Only first ten features 
    x = stack_imp["Feature"].values[0:10],
    mode='markers+text',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = stack_imp['Information Gain'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = stack_imp['Information Gain'].values[0:10]
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= None,
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
fig.update_layout(
    font=dict(size=25)
    )

fig.update_traces(textposition='top center')
pio.write_image(fig, "stack importance.png",scale=6, width=1500, height=1000)
"""

###### Explorative Data Analysis #####

sns.set(font_scale=1.65)
sns.pairplot(df_train_bi, hue = "class", hue_order = [1, 0], palette ='muted', vars = ["danceability", "instrumentalness", "acousticness"], height = 3, aspect=1.5)
plt.savefig("eda.png", dpi = 800, bbox_inches='tight',pad_inches = 0)


"""
# set plot style: grey grid in the background:
sns.set(style="darkgrid")
sns.set(font_scale=1.65)

# Set the figure size
plt.rcParams["figure.figsize"] = (21.5, 10)
# plot a bar chart
imp_plot = sns.barplot(
            x="Information Gain", 
            y="Feature", 
            data=imp.iloc[0:5,0:2], 
            estimator=sum, 
            errorbar=None, 
            color='#69b3a2')

for i in imp_plot.containers:
    imp_plot.bar_label(i,)

imp_plot.set(ylabel=None)    
plt.savefig("importance.png", dpi = 800, bbox_inches='tight',pad_inches = 0)
"""
complete_time = time.time() - complete_start 


