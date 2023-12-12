import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

# Suppress all warnings
warnings.simplefilter("ignore")

# Suppress all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

def count_moves(row, counts, index):
    total_moves = 0
    for i in range(1, 2564):
        move = row["Move "+ str(i)]
        # count the number of s's
        if move == 's':
            counts[10][index] += 1
        # count the number of Base's
        elif move == 'Base':
            counts[11][index] += 1
        # count the number of SingleMineral's
        elif move == 'SingleMineral':
            counts[12][index] += 1
        # count the hotkeys
        elif isinstance(move, str):
            for j in range(10):
                if move.startswith(f"hotkey{j}"):
                    counts[j][index] += 1

        total_moves += 1  
    # Save the total moves count
    counts[13][index] = total_moves


def count_move_per_time(row, counts, row_index, time_interval, ti_index):
    base_index = ti_index * 14
    total_moves = 0

    for i in range(1, 2564):
        move = row["Move " + str(i)]

        # Count actions for the given time interval
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

        # Continue counting actions after the specified time interval
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

# Drop unnecessary columns
train_data = train_data.drop(['PlayerURL', 'PlayerName'], axis=1)

# Create new table that only contains the first column (PlayerId) of train_data
# Keep only the first column but all rows
train_data_new = train_data.iloc[:, :1]


# Specify the target time intervals
#time_intervals = [20, 60, 100, 200]
time_intervals = [5, 20, 60, 100, 200, 270, 340, 550]

calc_column = len(time_intervals)* 14 + 14

# New lists of counts
counts = [[0] * 3052 for _ in range(calc_column)]
# New lists of races
races = [[0] * 3052 for _ in range(3)]


# Go through the rows using the functions to count the actions, map the races
for row_index, row in train_data.iterrows():
    count_moves(row, counts, row_index)
    mapRaces(races, row_index)

    for ti_index, time_interval in enumerate(time_intervals):
        count_move_per_time(row, counts, row_index, time_interval, ti_index+1)
        

for i in range(calc_column):
    locals()[f'count_{i}'] = counts[i]

for i in range(10):
    train_data_new[f'hk{i}Frequency'] = [count / counts[13][index] if counts[13][index] != 0 else 0 for index, count in enumerate(counts[i])]

train_data_new['sFrequency'] = [count / counts[13][index] if counts[13][index] != 0 else 0 for index, count in enumerate(counts[10])]
train_data_new['baseFrequency'] = [count / counts[13][index] if counts[13][index] != 0 else 0 for index, count in enumerate(counts[11])]
train_data_new['singleMineralFrequency'] = [count / counts[13][index] if counts[13][index] != 0 else 0 for index, count in enumerate(counts[12])]

# Adding new columns for the count of moves per interval
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

# Saving them in a csv file
train_data_new.to_csv('actiontype_count.csv', index=False)

print(train_data_new)

# Target
labels = train_data_new['PlayerID']

# Keep only the columns we need as features
features = train_data_new.drop(['PlayerID'], axis=1)

# Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)


# Choose a model (e.g., Decision Tree) and train it
model = RandomForestClassifier(random_state=42, n_estimators=200)
#model = GradientBoostingClassifier(random_state=42, n_estimators=200)

# Hyperparameter tuning using GridSearchCV
param_grid = {'n_estimators': [100, 150, 200], 'max_depth': [None, 10, 20]}
#grid_search = GridSearchCV(model, param_grid, cv=4)
grid_search = GridSearchCV(model, param_grid, cv=4, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_


# Choose the boosting algorithm (AdaBoost)
boosting_model = AdaBoostClassifier(best_model)

boosting_model.fit(X_train, y_train)


# Save the best model to a file
joblib.dump(boosting_model, 'player_id_prediction_model.pkl')

# Use the best model for predictions
predictions = boosting_model.predict(X_val)

print(f1_score(y_val, predictions, average='micro'))

# Use the best model for cross-validation scores
scores = cross_val_score(boosting_model, features, labels, cv=4)
print(scores)

# Evaluation of the model
accuracy = accuracy_score(y_val, predictions)
print(f'Accuracy: {accuracy}')

precision = precision_score(y_val, predictions, average='micro')
recall = recall_score(y_val, predictions, average='micro')

print(f'Precision: {precision}, Recall: {recall}')

# Explore feature importances
feature_importances = boosting_model.feature_importances_


# # Choose a model (e.g., Decision Tree) and train it
# #model = DecisionTreeClassifier(random_state=42)

# #model = RandomForestClassifier(random_state=42, n_estimators=100)
# #model.fit(X_train, y_train)

# # Trained model to a file saving
# joblib.dump(model, 'player_id_prediction_model.pkl')

# # Predictions on the val set
# #predictions = model.predict(X_val)

# print(f1_score(y_val,predictions,average='micro'))

# scores = cross_val_score(model, features, labels, cv=4)
# print(scores)

# # Evaluation of model
# accuracy = accuracy_score(y_val, predictions)
# print(f'Accuracy: {accuracy}')




# VISUALISATION

# Correlation Matrix
correlation_matrix = pd.DataFrame(X_train).corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.subplots_adjust(bottom=0.2)
plt.title("Correlation Heatmap", fontsize=14)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()

# Target Classes Distribution
plt.figure(figsize=(8, 6))
y_train.value_counts().plot(kind='bar', color="#98BF64")
plt.title("Distribution of Target Classes", fontsize=14)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.xlabel("PlayerID", fontsize=9)
plt.ylabel("Count", fontsize=9)
plt.show()

# Confusion Matrix (after model prediction)
conf_matrix = confusion_matrix(y_val, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Greens", fmt="d", cbar=False)
plt.title("Confusion Matrix", fontsize=14)
plt.xlabel("Predicted Label", fontsize=9)
plt.ylabel("True Label", fontsize=9)
plt.show()

# Feature Importance
feature_importances_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': best_model.feature_importances_})
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances_df, palette="crest")
plt.title("Feature Importance", fontsize=14)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.xlabel("Importance", fontsize=9)
plt.ylabel("Feature", fontsize=9)
plt.show()