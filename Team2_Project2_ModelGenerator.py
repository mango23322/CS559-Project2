import csv
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Sub A -- MLP Regressor
# Sub B -- Decision Tree

# To do:
# add more pre-processing
# Sub C -- (non-parametric model)
# Sub D -- (parametric model)
# Sub E --(stacking model using lin reg)


################# LOAD TRAIN DATA ###########################

x_raw = []
y_raw = []

# Upload training data as a list of lists.
csv_file = open("train.csv", "r")
train_data = list(csv.reader(csv_file, delimiter=","))
csv_file.close()

# Numericize and sort the data
for line in train_data:
    # Exclude the header line, any junk lines, and any bot results
    if line[0].isnumeric() and line[1][-3:]:
        # Store the game ID and score for this game
        x_raw.append( [int(line[0]), int(line[2])] )
        # Store the target (the rating for this score)
        y_raw.append(int(line[3]))
        
x_data = np.array(x_raw)
print('Training data ready')

################# LOAD TEST DATA ###########################

x_testraw = []
x_testIDs = []

csv_file = open("test.csv", "r")
test_data = list(csv.reader(csv_file, delimiter=","))
csv_file.close()

# Numericize and sort the data
for line in test_data:
    # Exclude the header line, any junk lines, and any bot results
    if line[3] == 'NA':
        # Store the game ID and score for this game
        x_testraw.append( [int(line[0]), int(line[2])] )
        # Separately store the game ID
        x_testIDs.append(int(line[0]))

x_test = np.array(x_testraw)
print('Test data ready')


################# PREPROCESSING ###########################
sc = StandardScaler()
x_data = sc.fit_transform(x_data)
x_test = sc.transform(x_test)


################# Sub A -- MLP Regressor ###########################

# This one takes 20 or 30 seconds on my PC
modelA = MLPRegressor(random_state=1, max_iter=500).fit(x_data, y_raw)
print('Model A built')
predictionA = modelA.predict(x_test)
# Create a submission out of the list of game IDs and the predictions for them
resultA = list(zip(x_testIDs, predictionA))


# Output a submission
    
filename = 'Team2_submissionA.csv'
csv_file = open(filename, "w", newline = '')
writerA = csv.writer(csv_file)
writerA.writerow(['game_id', 'rating'])
for row in resultA:
    writerA.writerow(row)

csv_file.close()

print('Submission complete!')


################# Sub B -- Decision Tree ###########################

# Create an instance of the model
modelB = DecisionTreeRegressor(random_state=0).fit(x_data, y_raw)
print('Model B built')
predictionB = modelB.predict(x_test)
# Create a submission
resultB = list(zip(x_testIDs, predictionB))

# Output a submission
    
filename = 'Team2_submissionB.csv'
csv_file = open(filename, "w", newline = '')
writerB = csv.writer(csv_file)
writerB.writerow(['game_id', 'rating'])
for row in resultB:
    writerB.writerow(row)

csv_file.close()

print('Submission complete!')


################# Sub C -- (non-parametric model) ###########################
## Anjali





################# Sub D -- (parametric model)  ###########################
## Kevin




################# Sub E --(stacking model using lin reg)  ###########################
## Jason