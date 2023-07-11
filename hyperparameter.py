import optuna 
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

###########################
### Objective Functions ###
###########################


class objective:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def xgb(self, trial):
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
        
        
        # Add a callback for pruning.
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-auc")
        
        clf=XGBClassifier(objective="binary:logistic", **space, eval_metric="auc" ,early_stopping_rounds=10, callbacks=[pruning_callback], random_state = 123)
        
        evaluation = [( self.x_train, self.y_train), ( self.x_test, self.y_test)]
        clf.fit(self.x_train, self.y_train,
                  eval_set=evaluation, verbose=False)
        preds = clf.predict(self.x_test)
        evalu = f1_score(self.y_test, preds)
        return evalu
    
    def logit(self, trial):
        space={"penalty": trial.suggest_categorical("penalty", [None, "elasticnet"]), 
                'C' : trial.suggest_float("C", 1e-2, 1, log = True),
                'l1_ratio' : trial.suggest_float("l1_ratio", 1e-2, 1)
            }
    

        
        clf = LogisticRegression(solver="saga", **space, max_iter = 10000, random_state = 123)        
        clf.fit(self.x_train, self.y_train)
        preds = clf.predict(self.x_test)
        evalu = f1_score(self.y_test, preds)
        return evalu
    
    def rf(self, trial): 
        space={'max_depth': trial.suggest_int("max_depth", 2, 12, 1),
                'criterion': trial.suggest_categorical('criterion', ["gini", "entropy", "log_loss"]),
                'max_features' : trial.suggest_categorical('max_features', ["sqrt", "log2", None]),
                'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 3),
                'min_samples_split' : trial.suggest_int('min_samples_split', 2, 4),
                'n_estimators': trial.suggest_int('n_estimators', 10, 300, 10)

            }
    
        clf=RandomForestClassifier(**space, bootstrap = False, random_state = 123)
        clf.fit(self.x_train, self.y_train,)
        preds = clf.predict(self.x_test)
        evalu = f1_score(self.y_test, preds)
        return evalu
    
    def gb(self, trial): 
        space={'max_depth': trial.suggest_int("max_depth", 2, 12, 1),
               'learning_rate': trial.suggest_float('learning_rate', 1e-2, 1, log = True),
                'criterion': trial.suggest_categorical('criterion', ["friedman_mse", "squared_error"]),
                'loss': trial.suggest_categorical('loss', ["log_loss", "exponential"]),
                'max_features' : trial.suggest_categorical('max_features', ["sqrt", "log2", None]),
                'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 3),
                'min_samples_split' : trial.suggest_int('min_samples_split', 2, 5),
                'n_estimators': trial.suggest_int('n_estimators', 10, 300, 10),
                "min_weight_fraction_leaf" : trial.suggest_float("min_weight_fraction_leaf", 1e-2, 0.5, log = True)

            }
    
        clf=GradientBoostingClassifier(**space, random_state = 123)
        clf.fit(self.x_train, self.y_train,)
        preds = clf.predict(self.x_test)
        evalu = f1_score(self.y_test, preds)
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
        clf.fit(self.x_train, self.y_train,)
        preds = clf.predict(self.x_test)
        evalu = f1_score(self.y_test, preds)
        return evalu

