import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
warnings.simplefilter("ignore")
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
# train_data_new.to_csv('actiontype_count.csv', index=False)

print(train_data_new)

# Target
labels = train_data_new['PlayerID']

# Keep only the columns we need as features
features = train_data_new.drop(['PlayerID'], axis=1)

# Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a model with all features
model = RandomForestClassifier(random_state=42, n_estimators=500)
model.fit(X_train, y_train)

# Get feature importances
feature_importances = model.feature_importances_

# Sort features based on importance
sorted_features = np.argsort(feature_importances)[::-1]

# Evaluate model performance for different feature subsets
best_accuracy = 0.0
best_feature_subset = None
eliminated_features = []

for num_features in range(1, len(features.columns) + 1):
    # Select the top N features based on importance
    selected_features = sorted_features[:num_features]
    eliminated_features.append(list(set(range(len(features.columns))) - set(selected_features)))

    X_train_subset = X_train.iloc[:, selected_features]
    X_val_subset = X_val.iloc[:, selected_features]

    # Train the model with the selected features
    model.fit(X_train_subset, y_train)

    # Make predictions on the validation set
    predictions = model.predict(X_val_subset)

    # Calculate accuracy
    accuracy = accuracy_score(y_val, predictions)

    # Check if this subset of features gives a better accuracy
    if accuracy >= best_accuracy:
        best_accuracy = accuracy
        best_feature_subset = selected_features.copy()


for num_features, eliminated in enumerate(eliminated_features):
    print(f"Features Eliminated with {num_features + 1}")
    print(f"{eliminated}")

# Select the best feature subset
final_features = X_train.iloc[:, best_feature_subset]

print(final_features)

# Save the final feature set to a file
#final_features.to_csv('final_features.csv', index=False)


# # Target
# labels = train_data_new['PlayerID']

# # Keep only the columns we need as features
# features = train_data_new.drop(['PlayerID'], axis=1)

# # Split the data into training and testing sets
# X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# # Train a model with all features
# model = RandomForestClassifier(random_state=42, n_estimators=500)
# model.fit(X_train, y_train)

# # Initialize variables
# best_accuracy = 93.45
# best_feature_subset = None
# current_percentage = 0.75
# step_size = 0.05  # Change the step size as needed
# eliminated_features = {}  # Dictionary to store eliminated features and their percentages

# while current_percentage >= 0.1:
#     # Calculate the number of features to keep
#     num_features_to_keep = int(len(features.columns) * current_percentage)

#     # Select the top N features based on importance
#     selected_features = model.feature_importances_.argsort()[-num_features_to_keep:][::-1]
#     eliminated_features[current_percentage] = list(set(range(len(features.columns))) - set(selected_features))

#     X_train_subset = X_train.iloc[:, selected_features]
#     X_val_subset = X_val.iloc[:, selected_features]

#     # Train the model with the selected features
#     model.fit(X_train_subset, y_train)

#     # Make predictions on the validation set
#     predictions = model.predict(X_val_subset)

#     # Calculate accuracy
#     accuracy = accuracy_score(y_val, predictions)

#     # Check if this subset of features gives a better accuracy
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_feature_subset = selected_features.copy()

#     # Update current percentage
#     current_percentage -= step_size

# # Select the best feature subset
# final_features = X_train.iloc[:, best_feature_subset]

# print("Best Accuracy:", best_accuracy)
# print("Number of Features Selected:", len(final_features.columns))

# # Print eliminated features and their percentages
# for percentage, eliminated in eliminated_features.items():
#     print(f"{eliminated}")
#     print(f"Features Eliminated {percentage * 100}%")

# Choose the boosting algorithm (AdaBoost)
boosting_model = AdaBoostClassifier(model)

boosting_model.fit(X_train, y_train)


# Save the best model to a file
joblib.dump(boosting_model, 'player_id_prediction_model.pkl')

# Use the best model for predictions
predictions = boosting_model.predict(X_val)

print(f1_score(y_val, predictions, average='micro'))

# Evaluation of the model
accuracy = accuracy_score(y_val, predictions)
print(f'Accuracy: {accuracy}')

precision = precision_score(y_val, predictions, average='micro')
recall = recall_score(y_val, predictions, average='micro')

print(f'Precision: {precision}, Recall: {recall}')

# Feature Importance
feature_importances_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': boosting_model.feature_importances_})
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances_df, palette="crest")
plt.title("Feature Importance", fontsize=12)
plt.xticks(fontsize=7)
plt.yticks(fontsize=5)
plt.xlabel("Importance", fontsize=9)
plt.ylabel("Feature", fontsize=9)
plt.show()

# Displaying the top correlated features
top_features = feature_importances_df.nlargest(15, 'Importance')['Feature']
correlation_matrix = pd.DataFrame(X_train[top_features]).corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Top Features)", fontsize=14)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()

# Target Classes Distribution
top_classes = y_train.value_counts().nlargest(30)
plt.figure(figsize=(12, 8))
top_classes.plot(kind='bar', color="#98BF64")
plt.title("Distribution of Top 30 Target Classes", fontsize=14)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.xlabel("PlayerID", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()

# Confusion Matrix (after model prediction)
top_classes_cm = y_train.value_counts().nlargest(50)

conf_matrix = confusion_matrix(y_val, predictions, labels=top_classes_cm.index)

plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", cbar=False)

plt.title("Confusion Matrix", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(fontsize=8, rotation=45, ha='right')
plt.yticks(fontsize=8)
plt.show()



