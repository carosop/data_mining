import pandas as pd
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









# model_filename = 'player_id_prediction_model.pkl'
# clf = joblib.load(model_filename)

# # Load and preprocess the data to be predicted
# test_data = pd.read_csv('test_data.csv', delimiter=';')
# test_data.columns = ['Race'] + [f'Move_{i}' for i in range(1, 3446)]

# print(test_data)
# # Map the 'Race' column to numeric values
# race_mapping = {'Protoss': 0, 'Zerg': 1, 'Terran': 2}
# test_data['Race'] = test_data['Race'].map(race_mapping)

# # Define action mapping with a default value of -1 for unrecognized actions
# action_mapping = {'s': 0, 'Base': 1, 'SingleMineral': 2}
# for i in range(10):
#     for j in range(3):
#         action_mapping[f'hotkey{i}{j}'] = 3 + i * 3 + j

# def map_action(action):
#     return action_mapping.get(action, -1)

# # Convert action sequences to numerical values 
# for i in range(1, 3446):
#     test_data[f'Move_{i}'] = test_data[f'Move_{i}'].map(map_action)

# print(test_data)

# # Define features (X) and target (y)
# features = ['Race'] + [f'Move_{i}' for i in range(1, 3446)]

# # Use the loaded model to make predictions
# predictions = clf.predict(test_data[features])

# print("Predicted Player IDs:")
# print(predictions)
