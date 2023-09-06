import numpy as np
import pandas as pd
import optuna 
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import copy



########################
### Cross Validation ###
########################

def crossval(model, X, y):
    """
    Custom function for cross validation, using sklearn StratifiedKFold as base and F1-Score as evaluation 
    Used in model_selector
    """
        
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state = 123) 
    cv_scores = np.empty(5)
    
    X = pd.DataFrame(X)
    y = pd.Series(y)
    mod = model

    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X1, X2 = X.iloc[train_idx], X.iloc[test_idx]
        y1, y2 = y[train_idx], y[test_idx]
        
        mod.fit(X1, y1)
        preds = mod.predict(X2)
        cv_scores[idx] = f1_score(y2, preds, average = "weighted")      
    return np.mean(cv_scores)



######################
### Model Stacking ###
######################
def train_oof_predictions(X, y, models):
    """
    Function to perform oof predictions on train data
    returns re-ordered predictors x, re-ordered target y, and model dictionary with filled predictors
    
    """
    
    # divide data into 5 folds
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state = 123)
    
    # prepare lists to hold the x and y values 
    data_x, data_y = [], []
    X_arr = np.array(X)
    y_arr = np.array(y)
    
    mod = copy.deepcopy(models) #deepcopy do avoid any problems
    
    # run the following block for each of the kfold splits
    for idx, (train_idx, test_idx) in enumerate(kfold.split(X_arr, y_arr)):
    
        #create this fold's training and test sets
        train_X, test_X = X_arr[train_idx], X_arr[test_idx] 
        train_y, test_y = y_arr[train_idx], y_arr[test_idx]
        
        # add the data that is used in this fold to lists
        data_x.extend(test_X)
        data_y.extend(test_y)
    
        # run each model on this fold and add the predictors to the model's running predictors list
        for item in mod:
            model = mod[item][0] # get the model to use on the fold
                            
            # fit and make predictions 
            model.fit(train_X, train_y) # fit to the train set for the kfold
            preds = model.predict(test_X) # fit on the remaining out-of-fold set
            mod[item][1].extend(preds) # add predictions to the model's running predictors list
            
    return data_x, data_y, mod


def model_selector(X, y, meta_model, models_dict, model_label):
    """
    Function to select the best base models for each meta-model
    Basic function in steps:
        1. Choose a meta-model
        2. For current meta_model, perform CV on original data and obtain baseline f1
        3. Add oof predictions for one of the base models to training data, re-fit meta_model and obtain updated F1 score of meta_model 
            -> do this for all base-models
        4. Compare updated F1-scores to baseline, add base-model whose updated accuracy was best to model stack
        5. In next round, again add oof predictions of base-models and see if accuracy of meta-model improves, add best model to stack
        6. Repeat 5. until accuracy no longer improves, then choose next meta_model and start from 2.
     """   
    
    included_models = []
     
    while True:
        changed=False
        
        # forward step
        
        excluded_models = list(set(models_dict.keys())-set(included_models)) # make a list of the current excluded_models
        new_acc = pd.Series(index=excluded_models, dtype=float) # make a series where the index is the current excluded_models
        
        current_meta_x = np.array(X)
        
        if len(included_models) > 0:
            for included in included_models:
                included = np.array(models_dict[included][1]).reshape((len(models_dict[included][1]), 1)) # gatheroof predictions of included models
                current_meta_x = np.hstack((current_meta_x, included)) # #add oof predictions of already included models to data stack

        starting_acc = round(crossval(meta_model, current_meta_x, y), 6)
       
        for excluded in excluded_models:  # for each item in the excluded_models list:
            
            new_yhat = np.array(models_dict[excluded][1]).reshape(-1, 1) # get the current models oof predictions
            meta_x = np.hstack((current_meta_x, new_yhat)) # add the predictions to the meta set
            
            # score the current item
            acc = round(crossval(meta_model, meta_x, y), 6)
            
            new_acc[excluded] = acc # append the f1 to the series field
        
        best_acc = new_acc.max() # evaluate best f1 of the excluded_models in this round
        
        if best_acc > starting_acc:  # if the best f1 is better than the initial f1
            best_feature = new_acc.idxmax()  # define best oof predictions as new best feature
            included_models.append(str(best_feature)) # append this model name to the included list
            changed=True # flag that change happend
        else: changed = False
        
        if not changed:
            break  # stacking no longer increases performance
            
    print(model_label, "model optimized")
    print("resulting models:", included_models)
    print("Accuracy:", starting_acc)
    
    return included_models, starting_acc



def create_meta_dataset(data_X, items):
    """
    Function that takes in a data set and list of predictions, and forges into one dataset
    """
    # Deepcopies to avoid changes in original data
    meta_x = copy.deepcopy(data_X)
    yhat_preds = copy.deepcopy(items)
    
    # combine prediction and data for each model
    for z in yhat_preds:
        z = np.array(z).reshape((len(z), 1))
        meta_x = np.hstack((meta_x, z))
    meta_x = pd.DataFrame(meta_x)
    meta_x.columns = np.arange(0, len(meta_x.columns))    
    return meta_x


def stack_prediction(X_test, final_models): 
    """
    Takes in a test set and a list of fitted models.
    Fits each model in the list on the test set and stores it in a predictions list. 
    Then uses create_meta_dataset to combine test and predictions 
    """
    predictions = []
    
    models = copy.deepcopy(final_models)
    X = copy.deepcopy(X_test)
    
    for model in models:
        preds = model.predict(X).reshape(-1,1) # make base models prediction for test set
        predictions.append(preds) 
    
    meta_X = create_meta_dataset(X, predictions)
        
    return meta_X



