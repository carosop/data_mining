# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# import joblib

# def count_moves(row, counts, index):
#     total_moves = 0
#     for i in range(1, 2564):
#         move = row["Move "+ str(i)]
#         # count the number of s's
#         if move == 's':
#             counts[10][index] += 1
#         # count the number of Base's
#         elif move == 'Base':
#             counts[11][index] += 1
#         # count the number of SingleMineral's
#         elif move == 'SingleMineral':
#             counts[12][index] += 1
#         # count the hotkeys
#         elif isinstance(move, str):
#             for j in range(10):
#                 if move.startswith(f"hotkey{j}"):
#                     counts[j][index] += 1

#         total_moves += 1  
#     # Save the total moves count
#     counts[13][index] = total_moves


# def count_move_per_time(row, counts, row_index, time_interval, ti_index):
#     base_index = ti_index * 14
#     total_moves = 0

#     for i in range(1, 2564):
#         move = row["Move " + str(i)]

#         # Count actions for the given time interval
#         if move == 's':
#             counts[base_index + 10][row_index] += 1
#         elif move == 'Base':
#             counts[base_index + 11][row_index] += 1
#         elif move == 'SingleMineral':
#             counts[base_index + 12][row_index] += 1
#         elif isinstance(move, str):
#             for j in range(10):
#                 if move.startswith(f"hotkey{j}"):
#                     counts[base_index + j][row_index] += 1

#         total_moves += 1

#         # Continue counting actions after the specified time interval
#         if move == f't{time_interval}':
#             break

#     counts[base_index + 13][row_index] = total_moves


# def mapRaces(races, row_index):
#     race = train_data['Race'][row_index]

#     if race == "Protoss":
#         races[0][row_index] = 1
#     elif race == "Terran":
#         races[1][row_index] = 1
#     elif race == "Zerg":
#         races[2][row_index] = 1

# # Load the training dataset
# train_data = pd.read_csv('train_data.csv', delimiter=';')

# # Drop unnecessary columns
# train_data = train_data.drop(['PlayerURL', 'PlayerName'], axis=1)

# # Create new table that only contains the first column (PlayerId) of train_data
# # Keep only the first column but all rows
# train_data_new = train_data.iloc[:, :1]


# # Specify the target time intervals
# #time_intervals = [20, 60, 100, 200]
# time_intervals = [5, 20, 60, 100, 200, 270, 340, 550]

# calc_column = len(time_intervals)* 14 + 14

# # New lists of counts
# counts = [[0] * 3052 for _ in range(calc_column)]
# # New lists of races
# races = [[0] * 3052 for _ in range(3)]


# # Go through the rows using the functions to count the actions, map the races
# for row_index, row in train_data.iterrows():
#     count_moves(row, counts, row_index)
#     mapRaces(races, row_index)

#     for ti_index, time_interval in enumerate(time_intervals):
#         count_move_per_time(row, counts, row_index, time_interval, ti_index+1)
        

# for i in range(calc_column):
#     locals()[f'count_{i}'] = counts[i]

# for i in range(10):
#     train_data_new[f'hk{i}Frequency'] = [count / counts[13][index] if counts[13][index] != 0 else 0 for index, count in enumerate(counts[i])]

# train_data_new['sFrequency'] = [count / counts[13][index] if counts[13][index] != 0 else 0 for index, count in enumerate(counts[10])]
# train_data_new['baseFrequency'] = [count / counts[13][index] if counts[13][index] != 0 else 0 for index, count in enumerate(counts[11])]
# train_data_new['singleMineralFrequency'] = [count / counts[13][index] if counts[13][index] != 0 else 0 for index, count in enumerate(counts[12])]

# # Adding new columns for the count of moves per interval
# for ti_index, time_interval in enumerate(time_intervals):
#     base_index = (ti_index + 1) * 14
#     for j in range(10):
#         column_name = f'hk{j}_t{time_interval}_Frequency'
#         train_data_new[column_name] = [count / counts[base_index + 13][index] if counts[base_index + 13][index] != 0 else 0 for index, count in enumerate(counts[base_index + j])]

#     train_data_new[f's_t{time_interval}_Frequency'] = [count / counts[base_index + 13][index] if counts[base_index + 13][index] != 0 else 0 for index, count in enumerate(counts[base_index + 10])]
#     train_data_new[f'base_t{time_interval}_Frequency'] = [count / counts[base_index + 13][index] if counts[base_index + 13][index] != 0 else 0 for index, count in enumerate(counts[base_index + 11])]
#     train_data_new[f'singleMineral_t{time_interval}_Frequency'] = [count / counts[base_index + 13][index] if counts[base_index + 13][index] != 0 else 0 for index, count in enumerate(counts[base_index + 12])]


# # Adding new columns for the races
# train_data_new['race_Protoss'] = races[0]
# train_data_new['race_Terran'] = races[1]
# train_data_new['race_Zerg'] = races[2]

# # Saving them in a csv file
# train_data_new.to_csv('actiontype_count.csv', index=False)

# print(train_data_new)


# # Salvataggio in un file CSV
# train_data_new.to_csv('actiontype_count.csv', index=False)

# # Target
# labels = train_data_new['PlayerID']

# # Mantieni solo le colonne necessarie come features
# features = train_data_new.drop(['PlayerID'], axis=1)

# # Divisione dei dati in set di addestramento e test
# X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# # Modello Random Forest
# model = RandomForestClassifier(random_state=42, n_estimators=200)

# # Ottimizzazione degli Iperparametri
# param_grid = {'n_estimators': [100, 150, 200, 250], 'max_depth': [None, 10, 20, 30]}
# grid_search = GridSearchCV(model, param_grid, cv=4, n_jobs=-1)
# grid_search.fit(X_train, y_train)
# best_model = grid_search.best_estimator_

# # Modello Gradient Boosting
# gradient_boosting_model = GradientBoostingClassifier(random_state=42, n_estimators=200)
# gradient_boosting_model.fit(X_train, y_train)

# # Modello di Ensemble (Voting Classifier)
# stacking_model = StackingClassifier(estimators=[('rf', best_model), ('gb', gradient_boosting_model)], final_estimator=LogisticRegression())
# stacking_model.fit(X_train, y_train)
# # voting_model = VotingClassifier(estimators=[('rf', best_model), ('gb', gradient_boosting_model)], voting='hard')
# # voting_model.fit(X_train, y_train)

# # Salvataggio dei modelli
# joblib.dump(best_model, 'random_forest_model.pkl')
# joblib.dump(gradient_boosting_model, 'gradient_boosting_model.pkl')
# joblib.dump(stacking_model, 'voting_model.pkl')
# # joblib.dump(voting_model, 'voting_model.pkl')

# print("-------------------------ended-----------------------------------")

# # Cross-Validation con Stratified K-Fold
# stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # # Cross-validation Random Forest
# # cross_val_scores_rf = cross_val_score(best_model, X_train, y_train, cv=stratified_kfold)
# # print("Cross-Validation Scores (RandomForest):", cross_val_scores_rf)

# # # Cross-validation Gradient Boosting
# # cross_val_scores_gb = cross_val_score(gradient_boosting_model, X_train, y_train, cv=stratified_kfold)
# # print("Cross-Validation Scores (GradientBoosting):", cross_val_scores_gb)

# # Cross-validation Voting Ensemble
# cross_val_scores_voting = cross_val_score(stacking_model, X_train, y_train, cv=stratified_kfold)
# print("Cross-Validation Scores (Voting Ensemble):", cross_val_scores_voting)





import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier
import numpy as np
import joblib
import warnings

warnings.simplefilter("ignore")

def count_moves(row, counts, index):
    total_moves = 0
    for i in range(1, 2564):
        move = row["Move " + str(i)]
        if move == 's':
            counts[10][index] += 1
        elif move == 'Base':
            counts[11][index] += 1
        elif move == 'SingleMineral':
            counts[12][index] += 1
        elif isinstance(move, str):
            for j in range(10):
                if move.startswith(f"hotkey{j}"):
                    counts[j][index] += 1

        total_moves += 1  
    counts[13][index] = total_moves

def count_move_per_time(row, counts, row_index, time_interval, ti_index):
    base_index = ti_index * 14
    total_moves = 0

    for i in range(1, 2564):
        move = row["Move " + str(i)]

        if move == 's':
            counts[base_index + 10][row_index] += 1
        elif move == 'Base':
            counts[base_index + 11][row_index] += 1
        elif move == 'SingleMineral':
            counts[base_index + 12][row_index] += 1
        elif isinstance(move, str):
            for j in range(10):
                if move.startswith(f"hotkey{j}"):
                    counts[base_index + j][row_index] += 1

        total_moves += 1

        if move == f't{time_interval}':
            break

    counts[base_index + 13][row_index] = total_moves

def mapRaces(races, row_index):
    race = train_data['Race'][row_index]

    if race == "Protoss":
        races[0][row_index] = 1
    elif race == "Terran":
        races[1][row_index] = 1
    elif race == "Zerg":
        races[2][row_index] = 1

# Load the training dataset
train_data = pd.read_csv('train_data.csv', delimiter=';')
train_data = train_data.drop(['PlayerURL', 'PlayerName'], axis=1)

# Create new table that only contains the first column (PlayerId) of train_data
train_data_new = train_data.iloc[:, :1]

# Specify the target time intervals
time_intervals = [5, 20, 60, 100, 200, 270, 340, 550]

calc_column = len(time_intervals) * 14 + 14
counts = [[0] * 3052 for _ in range(calc_column)]
races = [[0] * 3052 for _ in range(3)]

for row_index, row in train_data.iterrows():
    count_moves(row, counts, row_index)
    mapRaces(races, row_index)

    for ti_index, time_interval in enumerate(time_intervals):
        count_move_per_time(row, counts, row_index, time_interval, ti_index + 1)

for i in range(calc_column):
    locals()[f'count_{i}'] = counts[i]

for i in range(10):
    train_data_new[f'hk{i}Frequency'] = [count / counts[13][index] if counts[13][index] != 0 else 0 for index, count in enumerate(counts[i])]

train_data_new['sFrequency'] = [count / counts[13][index] if counts[13][index] != 0 else 0 for index, count in enumerate(counts[10])]
train_data_new['baseFrequency'] = [count / counts[13][index] if counts[13][index] != 0 else 0 for index, count in enumerate(counts[11])]
train_data_new['singleMineralFrequency'] = [count / counts[13][index] if counts[13][index] != 0 else 0 for index, count in enumerate(counts[12])]

for ti_index, time_interval in enumerate(time_intervals):
    base_index = (ti_index + 1) * 14
    for j in range(10):
        column_name = f'hk{j}_t{time_interval}_Frequency'
        train_data_new[column_name] = [count / counts[base_index + 13][index] if counts[base_index + 13][index] != 0 else 0 for index, count in enumerate(counts[base_index + j])]

    train_data_new[f's_t{time_interval}_Frequency'] = [count / counts[base_index + 13][index] if counts[base_index + 13][index] != 0 else 0 for index, count in enumerate(counts[base_index + 10])]
    train_data_new[f'base_t{time_interval}_Frequency'] = [count / counts[base_index + 13][index] if counts[base_index + 13][index] != 0 else 0 for index, count in enumerate(counts[base_index + 11])]
    train_data_new[f'singleMineral_t{time_interval}_Frequency'] = [count / counts[base_index + 13][index] if counts[base_index + 13][index] != 0 else 0 for index, count in enumerate(counts[base_index + 12])]

# Adding new columns for the races
train_data_new['race_Protoss'] = races[0]
train_data_new['race_Terran'] = races[1]
train_data_new['race_Zerg'] = races[2]

train_data_new.to_csv('actiontype_count.csv', index=False)

# Target
labels = train_data_new['PlayerID']

# Keep only the columns we need as features
features = train_data_new.drop(['PlayerID'], axis=1)

# Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Choose a model (e.g., Decision Tree) and train it
model = RandomForestClassifier(random_state=42, n_estimators=200)

# Hyperparameter tuning using GridSearchCV
param_grid = {'n_estimators': [100, 150, 200, 250], 'max_depth': [None, 10, 20, 30]}
grid_search = GridSearchCV(model, param_grid, cv=4, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Ensemble di Modelli: Combina RandomForest e AdaBoost
# rf_model = RandomForestClassifier(random_state=42, n_estimators=200)
ab_model = AdaBoostClassifier(best_model)
# ensemble_model = VotingClassifier(estimators=[('rf', best_model), ('ab', ab_model)], voting='hard')
ensemble_model = StackingClassifier(estimators=[('rf', best_model), ('ab', ab_model)],final_estimator=RandomForestClassifier(),stack_method='auto', cv=5 )
ensemble_model.fit(X_train, y_train)

# Save the best model to a file
joblib.dump(ensemble_model, 'player_id_prediction_model.pkl')

# Use the best model for predictions
predictions = ensemble_model.predict(X_val)

print(f1_score(y_val, predictions, average='micro'))

# Cross-Validation con Stratified K-Fold
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation Voting Ensemble
cross_val_scores_voting = cross_val_score(ensemble_model, X_train, y_train, cv=stratified_kfold)
print("Cross-Validation Scores (Voting Ensemble):", cross_val_scores_voting)


# Esplora l'importanza delle feature
# feature_importances = ensemble_model.feature_importances_
# print("Feature Importances:", feature_importances)
