import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib  # Import joblib for model saving

# Load the training dataset
train_data = pd.read_csv('train_data.csv', delimiter=';')
train_data.columns = ['PlayerURL', 'PlayerID', 'PlayerName', 'Race'] + [f'Move_{i}' for i in range(1, 2564)]

# Drop unnecessary columns
train_data = train_data.drop(['PlayerURL', 'PlayerName'], axis=1)

# Map race to numeric values
race_mapping = {'Protoss': 0, 'Zerg': 1, 'Terran': 2}
train_data['Race'] = train_data['Race'].map(race_mapping)

# Map actions to numeric values
action_mapping = {'s': 0, 'Base': 1, 'SingleMineral': 2}
for i in range(10):
    for j in range(3):
        action_mapping[f'hotkey{i}{j}'] = 3 + i * 3 + j

# Convert action sequences to numerical values
#if t it converts it to 100 
for i in range(1, 2564):
    train_data[f'Move_{i}'] = train_data[f'Move_{i}'].map(lambda x: 100 if pd.notna(x) and isinstance(x, str) and x.startswith('t') else action_mapping.get(x, -1))

#train_data.to_csv('mapped_train_data.csv', index=False)

print(train_data)


# Create an empty list to store counts for each row
row_action_counts = []

# Iterate through each row of the dataframe
for _, row in train_data.iterrows():
    # Initialize a counter for each time window
    action_count_before_time = 0

    # Initialize a dictionary to store counts for each time window
    counts_before_100 = {}

    count_100 = 1

    # Iterate through each 'Move_XX' column for the current row
    for col in train_data.columns[3:]:  # Assuming your 'Move_XX' columns start from index 3
        # Check if the value is different from -1
        if row[col] != -1:
            action_count_before_time += 1

            # Check if the value is 100
            if row[col] == 100:
                timestamp = count_100 * 5  # Use count_100 to calculate the timestamp
                counts_before_100[f't{timestamp}'] = action_count_before_time
                action_count_before_time = -1 
                count_100 += 1

    #If no action is found, set count to 0
    if not action_count_before_time:
        action_count_before_time = 0

    # Append the dictionary of counts for the current row to the main list
    row_action_counts.append(counts_before_100)

# Create a DataFrame from the results
result_df = pd.DataFrame(row_action_counts)

# Add the 'PlayerID' column
result_df.insert(0, 'PlayerID', train_data['PlayerID'])
result_df.insert(1, 'Race', train_data['Race'])

# Save the DataFrame to a CSV file
result_df.to_csv('move_count.csv', index=False)

# print("Move counts per player saved to 'move_count.csv'")


# # Print the results for each row
# for i, counts in enumerate(row_action_counts):
#     print(f"Row {i + 1}:\n{counts}")


# # Define features (X) and target (y)
# features = ['Race'] + [f'Move_{i}' for i in range(1, 2564)]
# target = 'PlayerID'

# # Create and train the Decision Tree classifier
# clf = DecisionTreeClassifier()
# clf.fit(train_data[features], train_data[target])

# # Save the trained model to a file
# model_filename = 'player_id_prediction_model.pkl'
# joblib.dump(clf, model_filename)

# print(f"Model saved as {model_filename}")
