import pandas as pd

# Load the training dataset
train_data = pd.read_csv('train_data.csv', delimiter=';')

# Drop unnecessary columns
train_data = train_data.drop(['PlayerURL', 'PlayerName'], axis=1)

# Create new table that only contains the first two columns (PlayerId and Race) of train_data
# Keep only the first two columns but all rows
train_data_new = train_data.iloc[:, :2]

# add the count of Moves per row

# new lists of counts
counts = [[0] * 3052 for _ in range(13)]

# go through the rows
for index, row in train_data.iterrows():

    # go through the moves 
    for i in range(1, 2564):
        
        # count the number of s's
        if row["Move "+ str(i)] == 's':
            counts[10][index] += 1
        
        # count the number of Base's
        elif row["Move "+ str(i)] == 'Base':
            counts[11][index] += 1
        
        # count the number of SingleMineral's
        elif row["Move "+ str(i)] == 'SingleMineral':
            counts[12][index] += 1
            
            
        # count the hotkeys
        elif isinstance(row["Move "+ str(i)], str):
            
            # count the number of hotkey0_'s
            if row["Move "+ str(i)].startswith("hotkey0"):
                counts[0][index] += 1
            
            # count the number of hotkey1_'s
            elif row["Move "+ str(i)].startswith("hotkey1"):
                counts[1][index] += 1
                
            # count the number of hotkey2_'s
            elif row["Move "+ str(i)].startswith("hotkey2"):
                counts[2][index] += 1
                
            # count the number of hotkey3_'s
            elif row["Move "+ str(i)].startswith("hotkey3"):
                counts[3][index] += 1
                
            # count the number of hotkey4_'s
            elif row["Move "+ str(i)].startswith("hotkey4"):
                counts[4][index] += 1
                
            # count the number of hotkey5_'s
            elif row["Move "+ str(i)].startswith("hotkey5"):
                counts[5][index] += 1
                
            # count the number of hotkey6_'s
            elif row["Move "+ str(i)].startswith("hotkey6"):
                counts[6][index] += 1
                
            # count the number of hotkey7_'s
            elif row["Move "+ str(i)].startswith("hotkey7"):
                counts[7][index] += 1
                
            # count the number of hotkey8_'s
            elif row["Move "+ str(i)].startswith("hotkey8"):
                counts[8][index] += 1
                
            # count the number of hotkey9_'s
            elif row["Move "+ str(i)].startswith("hotkey9"):
                counts[9][index] += 1
            
            

for i in range(10):
    train_data_new['hk' + str(i) + 'Counts'] = counts[i]

train_data_new['sCounts'] = counts[10]
train_data_new['baseCounts'] = counts[11]
train_data_new['singleMineralCounts'] = counts[12]
