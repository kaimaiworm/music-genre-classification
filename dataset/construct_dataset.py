import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
# current working directory
path = pathlib.Path().absolute()

##Genre coding
"""
0 = Accoustic/Folk
1 = Alternative Music
2 = Blues
3 = Bollywood
4 = Country
5 = HipHop
6 = Indie Music
7 = Instrumental
8 = Metal
9 = Pop
10 = Rock

"""

################
## Folders ##
################

input_folder         = "{}\\input".format(path)
output_folder        = "{}\\output".format(path)

################
### Functions ####
################

### Create functions for imputation of numeric values for missing within each class
# Imputation function
def impute(data):
    # extract numerical columns  
    cols = list(data)
    cols_remove = ["track name", "artist name", "class", "id"]
    cols = list(filter(lambda x: x not in cols_remove, cols))
    
    data_slice = data[cols].copy()
    min_val = np.zeros(14) 
    min_val[5] = -np.inf #define minimum values for imputer function
    imputer = IterativeImputer(max_iter=100, min_value = min_val)
    data_imp = imputer.fit_transform(data_slice)
    return data_imp, cols

# Function to iterate through classes and impute
def iterate_classes(data):
    for  genre in range(11):
         temp = data.loc[data["class"].isin([genre])].copy()
         temp_ID = list(temp["id"])
         temp_imp, cols = impute(temp)
         temp_imp = pd.DataFrame(temp_imp, columns = cols)
         temp_imp.insert(0, "id", temp_ID)
    
         for i in temp_ID:
             data.loc[data["id"].isin([i]), cols] = np.array(temp_imp.loc[temp_imp["id"].isin([i]), cols]) #this is not efficient, maybe fast way to assign imputations (e.g. through merge)
    return data


#######################
## Construct dataset ##
#######################


# Read in data of music songs
genres = pd.read_csv("{}\\train.csv".format(input_folder))

# Change names of columns to lowercase
genres.columns = [x.lower() for x in genres.columns]

# Create unique ID for each song
genres.insert(0, "id", pd.Series(range(1, len(genres) + 1)))

# Convert duration variable to seconds
genres.rename(columns={'duration_in min/ms':'duration'}, inplace=True)
genres["duration"].loc[genres["duration"] < 100] = genres["duration"].loc[genres["duration"] < 100]*60000


# Create training and test data sets (80:20)
train_set, test_set = train_test_split(genres, test_size = 0.2, random_state = 123)

train_set.reset_index(drop=True, inplace = True)
test_set.reset_index(drop=True, inplace = True)

# Impute values for test and training sets
train_imp = iterate_classes(train_set)
test_imp = iterate_classes(test_set)  

# Round imputed key values
train_imp["key"] = round(train_imp["key"])
test_imp["key"] = round(test_imp["key"])

# Turn categorical variables into binary dummies
enc = OneHotEncoder()
train_cat_encoded = pd.DataFrame(enc.fit_transform(train_imp[["key", "time_signature"]]).toarray(), columns = enc.get_feature_names_out())
test_cat_encoded = pd.DataFrame(enc.fit_transform(test_imp[["key", "time_signature"]]).toarray(), columns = enc.get_feature_names_out())

train_enc = pd.concat([train_imp, train_cat_encoded], axis = 1)
test_enc = pd.concat([test_imp, test_cat_encoded], axis = 1)
train_enc.drop(columns = ["key", "time_signature"], inplace=True)
test_enc.drop(columns = ["key", "time_signature"], inplace=True)

# Scale data
train_scaled = train_enc.copy()
test_scaled = test_enc.copy()
cols = ["popularity", "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", 
        "liveness", "valence", "tempo", "duration"]
scaler = StandardScaler()
train_scaled[cols] = pd.DataFrame(scaler.fit_transform(train_imp[cols]), columns=scaler.get_feature_names_out()) #only fit to training data to avoid data leakage
test_scaled[cols]  = pd.DataFrame(scaler.transform(test_imp[cols]), columns=scaler.get_feature_names_out())

#### Create new dataset with only two classes (Pop (1) and Other (0))
train_bi = train_scaled.copy()
test_bi = test_scaled.copy()

# Recode classes: Pop = 1, Other = 0
train_bi["class"] = np.where(train_bi["class"] == 9, 1, 0)
test_bi["class"] = np.where(test_bi["class"] == 9, 1, 0)

# Relative frequency of genres
print(genres["class"].value_counts(normalize=True))

# Save final dataset
with pd.ExcelWriter("{}\\genres_bi.xlsx".format(output_folder)) as writer:
    train_bi.to_excel(writer, sheet_name="train") 
    test_bi.to_excel(writer, sheet_name="test") 
    
# Save final dataset for multiclass classification
with pd.ExcelWriter("{}\\genres_multi.xlsx".format(output_folder)) as writer:
    train_scaled.to_excel(writer, sheet_name="train") 
    test_scaled.to_excel(writer, sheet_name="test") 



