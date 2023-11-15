import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib 

def count_moves(row, counts, index):
    for i in range(1, 3446):
        move = row["Move_"+ str(i)]

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

    for i in range(1, 3446):
        move = row["Move_"+ str(i)]
        
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


test_data = pd.read_csv('test_data.csv', delimiter=';')
test_data.columns = ['Race'] + [f'Move_{i}' for i in range(1, 3446)]


# Create new table that only contains the first two columns (PlayerId and Race) of train_data
# Keep only the first two columns but all rows
test_data_new = test_data.iloc[:, :1]

# add the count of Moves per row

# new lists of counts
counts = [[0] * 340 for _ in range(65)]

# Specify the target time intervals
time_intervals = [20, 60, 100, 200]

# go through the rows
for row_index, row in test_data.iterrows():
    count_moves(row, counts, row_index)

    for ti_index, time_interval in enumerate(time_intervals):
        count_move_per_time(row, counts, row_index, time_interval, ti_index+1)


for i in range(10):
    test_data_new[f'hk{i}Counts'] = counts[i]
    
test_data_new['sCounts'] = counts[10]
test_data_new['baseCounts'] = counts[11]
test_data_new['singleMineralCounts'] = counts[12]

for ti_index, time_interval in enumerate(time_intervals):
    base_index = (ti_index+1)*13
    for j in range(10):
        test_data_new[f'hk{j}_t{time_interval}_Counts'] = counts[base_index + j]

    test_data_new[f's_t{time_interval}_Counts'] = counts[base_index + 10]
    test_data_new[f'base_t{time_interval}_Counts'] = counts[base_index + 11]
    test_data_new[f'singleMineral_t{time_interval}_Counts'] = counts[base_index + 12]

test_data_new.to_csv('actiontype_count_test.csv', index=False)

print(test_data_new)


# Load the pre-trained model
model_filename = 'player_id_prediction_model.pkl'
clf = joblib.load(model_filename)

features = test_data_new.drop(['Race'], axis=1)


# Use the loaded model to make predictions
predictions = clf.predict(features)

# Add predictions to the test_data_new DataFrame
test_data_new['Predicted_PlayerID'] = predictions

print(test_data_new)


# Load the training dataset
train_data = pd.read_csv('train_data.csv', delimiter=';')

# Extract 'PlayerID' and 'PlayerURL' columns
player = train_data[['PlayerID', 'PlayerURL']]

player_info = player.drop_duplicates(subset='PlayerID', keep='first')

print(player_info)

# Save the result to CSV
player_info.to_csv('player_info.csv', index=False)


# Extract 'PlayerID' column
player_id_column = test_data_new[['Predicted_PlayerID']]

print(player_id_column)

# Merge the predicted ID to get the url of the player
result = pd.merge(player_id_column, player_info, left_on='Predicted_PlayerID', right_on='PlayerID', how='left')

result = result.drop(['Predicted_PlayerID', 'PlayerID'], axis=1)
result.insert(0,"Row_Id", range(1, len(result) + 1))

print(result)

# Save 'PlayerURL' to CSV
result.to_csv('player_id_only.csv', index=False)
