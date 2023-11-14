import pandas as pd

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


def count_move_per_time(row, counts, index, time_interval):
    found_time_interval = False

    for i in range(1, 2564):
        move = row["Move "+ str(i)]

        if found_time_interval:
            # Count actions for the given time interval
            if move == 's':
                counts[10][index] += 1
            elif move == 'Base':
                counts[11][index] += 1
            elif move == 'SingleMineral':
                counts[12][index] += 1

            # Count hotkeys for the given time interval
            elif isinstance(move, str):
                for j in range(10):
                    if move.startswith(f"hotkey{j}_t{time_interval}"):
                        counts[j][index] += 1

        # Check if the current action is the target time interval
        elif move == f't{time_interval}':
            found_time_interval = True

# Load the training dataset
train_data = pd.read_csv('train_data.csv', delimiter=';')

# Drop unnecessary columns
train_data = train_data.drop(['PlayerURL', 'PlayerName'], axis=1)

# Create new table that only contains the first two columns (PlayerId and Race) of train_data
# Keep only the first two columns but all rows
train_data_new = train_data.iloc[:, :2]

# add the count of Moves per row

# new lists of counts
counts = [[0] * 3052 for _ in range(68)]

# Specify the target time intervals
time_intervals = [20, 60, 100, 200]

# go through the rows
for index, row in train_data.iterrows():
    count_moves(row, counts, index)

    for i, time_interval in enumerate(time_intervals):
        count_move_per_time(row, counts, index, time_interval)


for i in range(10):
    train_data_new['hk' + str(i) + 'Counts'] = counts[i]

for i, time_interval in enumerate(time_intervals):
        base_index = 13 * i
        for j in range(10):
            train_data_new[f'hk{j}_t{time_interval}_Counts'] = counts[base_index + j]

        train_data_new[f's_t{time_interval}_Counts'] = counts[base_index + 10]
        train_data_new[f'base_t{time_interval}_Counts'] = counts[base_index + 11]
        train_data_new[f'singleMineral_t{time_interval}_Counts'] = counts[base_index + 12]

train_data_new['sCounts'] = counts[10]
train_data_new['baseCounts'] = counts[11]
train_data_new['singleMineralCounts'] = counts[12]

train_data_new.to_csv('actiontype_count.csv', index=False)

print(train_data_new)








# import pandas as pd

# def count_moves(row, counts, index):
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


# def count_move_per_time(row, counts, time_interval, count_time):
#     found_time_interval = False

#     for i in range(1, 2564):
#         move = row["Move "+ str(i)]

#         if found_time_interval:
#             for r in range(1, count_time):
#                 # Check if the current action is the target time interval
#                 if move == f't{time_interval}':
#                     found_time_interval = True
#                 # Count actions for the given time interval
#                 elif move == 's':
#                     counts[r*10] += 1
#                 elif move == 'Base':
#                     counts[r*11] += 1
#                 elif move == 'SingleMineral':
#                     counts[r*12] += 1

#                     # Count hotkeys for the given time interval
#                 elif isinstance(move, str):
#                     for j in range(10):
#                         if move.startswith(f"hotkey{j}_t{time_interval}"):
#                             counts[(r-1)*13 + j] += 1
#         elif move == f't{time_interval}':
#             found_time_interval = True


# # Load the training dataset
# train_data = pd.read_csv('train_data.csv', delimiter=';')

# # Drop unnecessary columns
# train_data = train_data.drop(['PlayerURL', 'PlayerName'], axis=1)

# # Create new table that only contains the first two columns (PlayerId and Race) of train_data
# # Keep only the first two columns but all rows
# train_data_new = train_data.iloc[:, :2]

# # add the count of Moves per row

# # new lists of counts
# counts = [[0] * 3052 for _ in range(68)]

# # Specify the target time intervals
# time_intervals = [20, 60, 100, 200]

# count_time = 4
# # go through the rows
# for index, row in train_data.iterrows():
#     count_moves(row, counts, index)

#     for i, time_interval in enumerate(time_intervals):
#         count_move_per_time(row, counts, time_interval, count_time)


# for i in range(10):
#     train_data_new['hk' + str(i) + 'Counts'] = counts[i]

# for time_interval in enumerate(time_intervals):
#     for i in range(1, count_time ):
#         base_index = 13 * i
#         for j in range(10):
#             train_data_new[f'hk{j}_t{time_interval}_Counts'] = counts[base_index + j]

#         train_data_new[f's_t{time_interval}_Counts'] = counts[base_index + 10]
#         train_data_new[f'base_t{time_interval}_Counts'] = counts[base_index + 11]
#         train_data_new[f'singleMineral_t{time_interval}_Counts'] = counts[base_index + 12]

# train_data_new['sCounts'] = counts[10]
# train_data_new['baseCounts'] = counts[11]
# train_data_new['singleMineralCounts'] = counts[12]

# train_data_new.to_csv('actiontype_count.csv', index=False)

# print(train_data_new)