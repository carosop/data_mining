import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib 

def count_moves(row, counts, index):
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


def count_move_per_time(row, counts, row_index, time_interval, ti_index):
    base_index = ti_index*13

    for i in range(1, 2564):
        move = row["Move "+ str(i)]
        
        if move == f't{time_interval}':
            return
        
        else:
            # Count actions for the given time interval
            if move == 's':
                counts[base_index+10][row_index] += 1
            elif move == 'Base':
                counts[base_index+11][row_index] += 1
            elif move == 'SingleMineral':
                counts[base_index+12][row_index] += 1

            # Count hotkeys for the given time interval
            elif isinstance(move, str):
                for j in range(10):
                    if move.startswith(f"hotkey{j}_t{time_interval}"):
                        counts[base_index+j][row_index] += 1


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

# Add columns for the count of moves per row
# Add columns for the count of moves per time interval
# Add columns for the races

# New lists of counts
counts = [[0] * 3052 for _ in range(65)]
# New lists of races
races = [[0] * 3052 for _ in range(3)]

# Specify the target time intervals
time_intervals = [20, 60, 100, 200]

# Go through the rows using the functions to count the actions, map the races
for row_index, row in train_data.iterrows():
    count_moves(row, counts, row_index)
    mapRaces(races, row_index)

    for ti_index, time_interval in enumerate(time_intervals):
        count_move_per_time(row, counts, row_index, time_interval, ti_index+1)
        
# Adding all the new columns to the train_data_new
# Adding new columns for the count of moves
for i in range(10):
    train_data_new[f'hk{i}Counts'] = counts[i]
    
train_data_new['sCounts'] = counts[10]
train_data_new['baseCounts'] = counts[11]
train_data_new['singleMineralCounts'] = counts[12]

# Adding new columns for the count of moves per interval
for ti_index, time_interval in enumerate(time_intervals):
    base_index = (ti_index+1)*13
    for j in range(10):
        train_data_new[f'hk{j}_t{time_interval}_Counts'] = counts[base_index + j]

    train_data_new[f's_t{time_interval}_Counts'] = counts[base_index + 10]
    train_data_new[f'base_t{time_interval}_Counts'] = counts[base_index + 11]
    train_data_new[f'singleMineral_t{time_interval}_Counts'] = counts[base_index + 12]

# Adding new columns for the races
train_data_new['race_Protoss'] = races[0]
train_data_new['race_Terran'] = races[1]
train_data_new['race_Zerg'] = races[2]

# Saving them in a csv file
train_data_new.to_csv('actiontype_count.csv', index=False)

print(train_data_new)


# Target
labels = train_data_new['PlayerID']

# Keep only the colums we need as features
features = train_data_new.drop(['PlayerID'], axis=1)

# Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Choose a model (e.g., Decision Tree) and train it
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Trained model to a file saving
joblib.dump(model, 'player_id_prediction_model.pkl')

# Predictions on the val set
predictions = model.predict(X_val)

print(f1_score(y_val,predictions,average='micro'))

scores = cross_val_score(model, features, labels, cv=3)
print(scores)

# Evaluation of model
accuracy = accuracy_score(y_val, predictions)
print(f'Accuracy: {accuracy}')
