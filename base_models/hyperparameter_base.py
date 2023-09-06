import optuna 
import numpy as np
import optuna 
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import time


        
###########################
### Objective Functions ###
###########################

"""
Defines the objective functions for each of the ML models for hyperparameter tuning using optuna, 
i.e. sample parameter space, fitting model and returning F1 score

"""

class objective:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def gbt(self, trial):
        space={'max_depth': trial.suggest_int("max_depth", 3, 18, 1),
                'gamma': trial.suggest_float('gamma', 1,9),
                'alpha' : trial.suggest_int('alpha', 20,180,1),
                'lambda' : trial.suggest_float('lambda', 1e-2, 1),
                'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1, log = True),
                'min_child_weight' : trial.suggest_int('min_child_weight', 0, 10, 1),
                'learning_rate': trial.suggest_float('learning_rate', 1e-2, 1, log = True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, 100),
                'seed': 0
            }
        
     
        clf=XGBClassifier(objective="binary:logistic", **space, n_jobs = -1, random_state = 123)
        X1, X2, y1, y2 = train_test_split(self.X, self.y, test_size = 0.2, random_state=123)        
        clf.fit(X1, y1)
        preds = clf.predict(X2)
        evalu = f1_score(y2, preds, average = "binary")
        return evalu
    
    def logit(self, trial):
        space={'C' : trial.suggest_float("C", 1e-2, 1, log = True),
                'l1_ratio' : trial.suggest_float("l1_ratio", 1e-2, 1)
            }
    
        clf = LogisticRegression(solver="saga", penalty = "elasticnet", **space, max_iter = 10000, random_state = 123)        
        X1, X2, y1, y2 = train_test_split(self.X, self.y, test_size = 0.2, random_state=123)
        clf.fit(X1, y1)
        preds = clf.predict(X2)
        evalu = f1_score(y2, preds)
        return evalu
    
    def rf(self, trial): 
        space={'max_depth': trial.suggest_int("max_depth", 2, 12, 1),
                'criterion': trial.suggest_categorical('criterion', ["gini", "entropy", "log_loss"]),
                'max_features' : trial.suggest_categorical('max_features', ["sqrt", "log2", None]),
                'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 3),
                'min_samples_split' : trial.suggest_int('min_samples_split', 2, 4),
                'n_estimators': trial.suggest_int('n_estimators', 10, 300, 10)

            }
    
        clf=RandomForestClassifier(**space, random_state = 123, n_jobs = -1)
        X1, X2, y1, y2 = train_test_split(self.X, self.y, test_size = 0.2, random_state=123)
        clf.fit(X1, y1)
        preds = clf.predict(X2)
        evalu = f1_score(y2, preds, average = "binary")
        return evalu

    
    def dt(self, trial):
        space={'max_depth': trial.suggest_int("max_depth", 2, 12, 1),
                'criterion': trial.suggest_categorical('criterion', ["gini", "entropy", "log_loss"]),
                'max_features' : trial.suggest_categorical('max_features', ["sqrt", "log2", None]),
                'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 3),
                'min_samples_split' : trial.suggest_int('min_samples_split', 2, 6),
                'splitter': trial.suggest_categorical('splitter', ["best", "random"])

            }
    
        clf=DecisionTreeClassifier(**space, random_state = 123)
        X1, X2, y1, y2 = train_test_split(self.X, self.y, test_size = 0.2, random_state=123)
        clf.fit(X1, y1)
        preds = clf.predict(X2)
        evalu = f1_score(y2, preds, average = "binary")
        return evalu

    def knn(self, trial):
        space={'n_neighbors': trial.suggest_int("n_neighbors", 2, 12, 1),
                'weights': trial.suggest_categorical('weights', ["uniform", "distance"]),
                'algorithm' : trial.suggest_categorical('algorithm', ["ball_tree", "kd_tree"]),
                'leaf_size' : trial.suggest_int('leaf_size', 20, 50, 1),
                'p' : trial.suggest_int('p', 1, 3)
            }
    
        clf=KNeighborsClassifier(**space, n_jobs = -1)
        X1, X2, y1, y2 = train_test_split(self.X, self.y, test_size = 0.2, random_state=42)
        clf.fit(X1, y1)
        preds = clf.predict(X2)
        evalu = f1_score(y2, preds, average = "binary")
        return evalu
    
    def svm(self, trial):
        space={'kernel': trial.suggest_categorical('kernel', ["linear", "rbf"]),
                'gamma' : trial.suggest_categorical('gamma', ["scale", "auto"]),
                'C' : trial.suggest_float('C', 0, 1)
            }
    
        clf=SVC(**space, probability = True, random_state = 123)
        X1, X2, y1, y2 = train_test_split(self.X, self.y, test_size = 0.2, random_state=123)
        clf.fit(X1, y1)
        preds = clf.predict(X2)
        evalu = f1_score(y2, preds, average = "binary")
        return evalu
    
    def lda(self, trial):
        space={'shrinkage': trial.suggest_categorical('shrinkage', ["auto", None])
            }
    
        clf=LinearDiscriminantAnalysis(solver = "lsqr", **space)
        X1, X2, y1, y2 = train_test_split(self.X, self.y, test_size = 0.2, random_state=123)
        clf.fit(X1, y1)
        preds = clf.predict(X2)
        evalu = f1_score(y2, preds, average = "binary")
        return evalu
    
    def qda(self, trial):
        space={'reg_param': trial.suggest_float("reg_param", 0,1)
            }
    
        clf=QuadraticDiscriminantAnalysis(**space)
        X1, X2, y1, y2 = train_test_split(self.X, self.y, test_size = 0.2, random_state=123)
        clf.fit(X1, y1)
        preds = clf.predict(X2)
        evalu = f1_score(y2, preds, average = "binary")
        return evalu




def tuning(objective_function, models):
    """
    Helper function for hyperparameter tuning using optuna, calls hyperparameter_base.objective()
    """
    # init dicts
    study_dict = {}
    best_params = {}
    timer = {}    
    #iterate through list of models
    for mod in models:
        np.random.seed(42) #seed for reproducability
        start_time = time.time() #start timing for each model

        #create study
        study_dict[mod] = optuna.create_study(direction="maximize",
                                              study_name = "study_{}".format(mod),
                                              sampler=optuna.samplers.TPESampler(seed=42),
                                              pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
        study_dict[mod].optimize(getattr(objective_function, str(mod)), n_trials=100) # start optimization for current model
        best_params[mod] = study_dict[mod].best_params #save best pararms for current model
        end_time = time.time() - start_time
        timer[mod] = end_time #save time needed for hyperparameter tuning
    return best_params, timer



    

