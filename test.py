import pandas as pd
import joblib 

model_filename = 'player_id_prediction_model.pkl'
clf = joblib.load(model_filename)

# Load and preprocess the data to be predicted
test_data = pd.read_csv('test_data.csv', delimiter=';')
test_data.columns = ['Race'] + [f'Move_{i}' for i in range(1, 3446)]

print(test_data)
# Map the 'Race' column to numeric values
race_mapping = {'Protoss': 0, 'Zerg': 1, 'Terran': 2}
test_data['Race'] = test_data['Race'].map(race_mapping)

# Define action mapping with a default value of -1 for unrecognized actions
action_mapping = {'s': 0, 'Base': 1, 'SingleMineral': 2}
for i in range(10):
    for j in range(3):
        action_mapping[f'hotkey{i}{j}'] = 3 + i * 3 + j

def map_action(action):
    return action_mapping.get(action, -1)

# Convert action sequences to numerical values 
for i in range(1, 3446):
    test_data[f'Move_{i}'] = test_data[f'Move_{i}'].map(map_action)

print(test_data)

# Define features (X) and target (y)
features = ['Race'] + [f'Move_{i}' for i in range(1, 3446)]

# Use the loaded model to make predictions
predictions = clf.predict(test_data[features])

print("Predicted Player IDs:")
print(predictions)
