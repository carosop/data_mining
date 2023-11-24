import pandas as pd
from itertools import groupby, combinations

###################################################
# Preprocessing
###################################################
# Load the training dataset
train_data = pd.read_csv('train_data.csv', delimiter=';')
train_data.columns = ['PlayerURL', 'PlayerID', 'PlayerName', 'Race'] + [f'Move_{i}' for i in range(1, 2564)]

# Drop unnecessary columns
train_data = train_data.drop(['PlayerURL', 'PlayerName'], axis=1)

# Get the indices for columns 'Move_1' to 'Move_50'
move_columns = [f'Move_{i}' for i in range(1, 68)]

data_10s = []

# Iterate through each row of the dataframe
for _, row in train_data.iterrows():
    row_actions = []

    # Iterate through each 'Move_XX' column for the current row
    for col in train_data.columns[3:70]:
        row_actions.append(row[col])

        # Check if the current action is 't10'
        if row[col] == 't10':  # Assuming 't10' is converted to 100 in your previous processing
            break  # Stop iterating if 't10' is found

    data_10s.append(row_actions)

# Convert the result to a new dataframe if needed
data_10s_df = pd.DataFrame(data_10s, columns=move_columns)

data_10s_df.insert(0, 'PlayerID', train_data['PlayerID'])
data_10s_df.insert(1, 'Race', train_data['Race'])

data_10s_df.to_csv('data_10s.csv', index=False)

# Load the data_10s.csv file
data_10s_df = pd.read_csv('data_10s.csv')

# Get the move columns
move_columns = [f'Move_{i}' for i in range(1, 68)]

# Flatten the dataframe to have a single column of moves
all_moves = data_10s_df[move_columns].values.flatten()

# Count the frequency of each move
moves_frequency = {}
for move in all_moves:
    if pd.notna(move):  # Exclude NaN values
        moves_frequency[move] = moves_frequency.get(move, 0) + 1

print("Moves Frequency:")
for move, frequency in moves_frequency.items():
    print(f"{move}: {frequency}")

# Group the data by PlayerID and reset the index
grouped_data = data_10s_df.groupby('PlayerID')[move_columns].apply(lambda x: x.reset_index(drop=True))

# Define a function to find sequences of consecutive moves
def find_sequences(group):
    sequences = []
    for _, g in groupby(enumerate(group), key=lambda x: int(x[1] == 't10')):
        consecutive_moves = list(map(lambda x: x[1], g))

        # Remove 't5' and 't10' from consecutive_moves
        consecutive_moves = [move for move in consecutive_moves if move not in ['t5', 't10']]

        sequences.append(consecutive_moves)
    return sequences

# Iterate through each player's moves and find sequences
player_sequences = {}
for player, moves in grouped_data.iterrows():
    sequences = find_sequences(moves.dropna().astype(str))
    player_sequences[player] = sequences

# Save the found sequences to a text file
output_file_path = 'sequences.txt'
with open(output_file_path, 'w') as file:
    for player, sequences in player_sequences.items():
        file.write(f"{player} : \t")
        for sequence in sequences:
            file.write(','.join(sequence) + '\n')
        file.write("\n")

print(f"Sequences saved to {output_file_path}")

# Define a function to evaluate the combination of moves
def evaluate_combination(combination):
    unique_moves = set(combination[0] + combination[1])
    uniqueness_score = sum(moves_frequency.get(move, 0) for move in unique_moves)
    return uniqueness_score

# Iterate through each player's sequences and find ranked combinations
ranked_combinations = {}
for player, sequences in player_sequences.items():
    combinations_list = list(combinations(sequences, 2))
    ranked_combinations[player] = sorted(combinations_list, key=lambda x: evaluate_combination(x), reverse=True)

# Print or save the ranked combinations
for player, combinations in ranked_combinations.items():
    print(f"Player {player} ranked combinations:")
    for i, combination in enumerate(combinations, start=1):
        print(f"Rank {i}: {combination} - Score: {evaluate_combination(combination)}")

# Save the ranked combinations to a text file
output_file_path = 'ranked_combinations.txt'
with open(output_file_path, 'w') as file:
    for player, combinations in ranked_combinations.items():
        file.write(f"Player {player} ranked combinations:\n")
        for i, combination in enumerate(combinations, start=1):
            file.write(f"Rank {i}: {combination} - Score: {evaluate_combination(combination)}\n")
        file.write("\n")

print(f"Ranked combinations saved to {output_file_path}")






# # Map race to numeric values
# race_mapping = {'Protoss': 0, 'Zerg': 1, 'Terran': 2}
# train_data['Race'] = train_data['Race'].map(race_mapping)

# # Map actions to numeric values
# action_mapping = {'s': 0, 'Base': 1, 'SingleMineral': 2}
# for i in range(10):
#     for j in range(3):
#         action_mapping[f'hotkey{i}{j}'] = 3 + i * 3 + j

# # Convert action sequences to numerical values
# #if tXX it converts it to 100 otherwise -1
# for i in range(1, 2564):
#     train_data[f'Move_{i}'] = train_data[f'Move_{i}'].map(lambda x: 100 if pd.notna(x) and isinstance(x, str) and x.startswith('t') else action_mapping.get(x, -1))


# # Count how many types of races each player plays
# race_counts = train_data.groupby('PlayerID')['Race'].nunique().reset_index(name='NumRaces')
# # Count how many times each player plays each race
# race_occurrences = train_data.groupby(['PlayerID', 'Race']).size().reset_index(name='NumOccurrences')

# # Merge the two DataFrames on 'PlayerID'
# merge = pd.merge(race_counts, race_occurrences, on='PlayerID', how='left')

# # Save the result to a CSV file
# merge.to_csv('race_count_per_player.csv', index=False)

# print(merge)

# #train_data.insert(1,'NumRaces', race_counts['NumRaces'])

# train_data.to_csv('mapped_train_data.csv', index=False)


# print(train_data)


# ########################################################
# # counting of how many actions before each time slot t
# ########################################################
# # Create an empty list to store counts for each row
# row_action_counts = []

# # Iterate through each row of the dataframe
# for _, row in train_data.iterrows():
#     # Initialize a counter for each time window
#     action_count_before_time = 0

#     # Initialize a dictionary to store counts for each time window
#     counts_before_100 = {}

#     count_100 = 1

#     # Iterate through each 'Move_XX' column for the current row
#     for col in train_data.columns[3:]: 
#         # Check if the value is different from -1
#         if row[col] != -1:
#             action_count_before_time += 1

#             # Check if the value is 100
#             if row[col] == 100:
#                 timestamp = count_100 * 5 
#                 counts_before_100[f't{timestamp}'] = action_count_before_time
#                 action_count_before_time = -1 
#                 count_100 += 1

#     #If no action is found, set count to 0
#     if not action_count_before_time:
#         action_count_before_time = 0

#     # Append the counts for the current row to the list
#     row_action_counts.append(counts_before_100)

# # Create a csv from the results
# result_df = pd.DataFrame(row_action_counts)

# # Add the 'PlayerID' and 'Race' column
# result_df.insert(0, 'PlayerID', train_data['PlayerID'])
# result_df.insert(1, 'Race', train_data['Race'])

# # Save the DataFrame to a CSV file
# result_df.to_csv('move_count.csv', index=False)

# # # Print the results for each row
# # for i, counts in enumerate(row_action_counts):
# #     print(f"Row {i + 1}:\n{counts}")


# # # Define features (X) and target (y)
# # features = ['Race'] + [f'Move_{i}' for i in range(1, 2564)]
# # target = 'PlayerID'

# # # Create and train the Decision Tree classifier
# # clf = DecisionTreeClassifier()
# # clf.fit(train_data[features], train_data[target])

# # # Save the trained model to a file
# # model_filename = 'player_id_prediction_model.pkl'
# # joblib.dump(clf, model_filename)

# # print(f"Model saved as {model_filename}")
