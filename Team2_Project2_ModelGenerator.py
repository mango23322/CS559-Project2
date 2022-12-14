import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

# Sub A -- MLP Regressor
# Sub B -- Decision Tree
# Sub C -- KNN
# Sub D -- Linear Regression
# Sub E --(stacking model using lin reg)


################# LOAD TRAIN DATA ###########################

games = pd.read_csv('games.csv')
turns = pd.read_csv('turns.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

x_train_raw = []
y_train_raw = []

# Upload training data as a list of lists.
csv_file = open("train.csv", "r")
train_data = list(csv.reader(csv_file, delimiter=","))
csv_file.close()

# Numericize and sort the data
for line in train_data:
    # Exclude the header line, any junk lines, and any bot results
    if line[0].isnumeric() and line[1][-3:]:
        # Store the game ID and score for this game
        x_train_raw.append( [int(line[0]), int(line[2])] )
        # Store the target (the rating for this score)
        y_train_raw.append(int(line[3]))
        
x_train = np.array(x_train_raw)
y_train = np.array(y_train_raw)
print('Training data ready')

################# LOAD TEST DATA ###########################

x_test_raw = []
x_test_IDs_raw = []

csv_file = open("test.csv", "r")
test_data = list(csv.reader(csv_file, delimiter=","))
csv_file.close()

# Numericize and sort the data
for line in test_data:
    # Exclude the header line, any junk lines, and any bot results
    if line[3] == 'NA':
        # Store the game ID and score for this game
        x_test_raw.append( [int(line[0]), int(line[2])] )
        # Separately store the game ID
        x_test_IDs_raw.append(int(line[0]))

x_test = np.array(x_test_raw)
x_test_IDs = np.array(x_test_IDs_raw)
print('Test data ready')

################# PREPROCESSING ###########################

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

## games.csv file ##
#This file has 4 columns of categorical data, which are time_control_name, game_end_reason, lexicon, rating_mode.
#Encoding all these 4 columns for better prediction using LabelEncoder module
le = preprocessing.LabelEncoder()
games['time_control_name'] = le.fit_transform(games.time_control_name.values)
games['game_end_reason'] = le.fit_transform(games.game_end_reason.values)
games['lexicon'] = le.fit_transform(games.lexicon.values)
games['rating_mode'] = le.fit_transform(games.rating_mode.values)
#dropping the columns created_at, increment_seconds, winner
games = games.drop(['created_at', 'increment_seconds'],axis=1)
# print(games.head(15))

## turns.csv file ##
#dropping the columns rack, location, move, points
#print(turns.corr())
turns = turns.drop(['rack','location','move','points','score','nickname','turn_type'],axis=1)
#grouping the column together on basis of game_id, to calculate the total number of turn_number for each game_id
new_turns = turns.groupby(pd.Grouper(key='game_id', axis=0, sort=False )).max() #grouping individual game_id's together
# new_turns.head()

## train.csv file ##
#creating a correlation matrix to see dependency between features 
# train.corr()

############################ RMSE ##################################
#Separating the rows for which nickname ends with 'bot', to calcualte rmse
bot_data = train[train.nickname.str.endswith('Bot')].copy(deep=True)
bot_data['nickname'] = le.fit_transform(bot_data.nickname.values)

#merging the new_turn and bot_data on basis of game_id
merged_data = pd.merge(new_turns, bot_data, on='game_id')
#print(merged_data)

#The turn_number contains a collective number of turns played by both human player and bot, inorder to get an approximate turn_number of bot only we need to divide it by 2.
#But as for all the game_ids there is one  turn_number entry to specify the winner, subtracting 1 from the turn_number before dividing it by 2
merged_data['turn_number'] = ((merged_data['turn_number'])-1)/2
merged_data['turn_number'] = merged_data['turn_number'].astype('int')
#print(merged_data)

#merging the merged_data and games data on the basis of game_id
new_merged = pd.merge(merged_data, games, on='game_id')
new_merged = new_merged.drop(['first','max_overtime_minutes'], axis =1)
# new_merged.head()

## Train data ##
x_bot_train = new_merged.drop('rating',axis=1)
y_bot_train = new_merged['rating'].values.reshape(-1,1)

################# Sub A -- MLP Regressor ###########################

# Parametric model
def my_MLPRegressor(x_train, y_train, x_test, x_test_IDs):
    # This one takes 20 or 30 seconds on my PC
    modelA = MLPRegressor(random_state=1, max_iter=500).fit(x_train, y_train)
    score_A = modelA.score(x_train, y_train)
    print("Score of model A:", score_A)
    predictionA = modelA.predict(x_test)
    print('Model A built')

    # rmse_train_mlp=mean_squared_error(y_raw, predictionA,squared=False)
    # print("RMSE of MLPRegressor:",rmse_train_mlp)
    
    # Create a submission out of the list of game IDs and the predictions for them
    resultA = list(zip(x_test_IDs, predictionA))

    # Output a submission        
    filename = 'Team2_submissionA.csv'
    csv_file = open(filename, "w", newline='')
    writerA = csv.writer(csv_file)
    writerA.writerow(['game_id', 'rating'])
    for row in resultA:
        writerA.writerow(row)
    csv_file.close()
    print('Submission A complete!')

    return modelA

# ################# Sub B -- Decision Tree ###########################

# Non-parametric model
def my_DecisionTreeRegressor(x_train, y_train, x_test, x_test_IDs):
    # Create an instance of the model
    modelB = DecisionTreeRegressor(random_state=0).fit(x_train, y_train)
    score_B = modelB.score(x_train, y_train)
    print("Score of model B:", score_B)
    predictionB = modelB.predict(x_test)
    print('Model B built')

    # rmse_train_dt=mean_squared_error(y_raw, predictionB,squared=False)
    # print("RMSE of Decision tree is:",rmse_train_dt)

    # Create a submission
    resultB = list(zip(x_test_IDs, predictionB))

    # Output a submission
    filename = 'Team2_submissionB.csv'
    csv_file = open(filename, "w", newline = '')
    writerB = csv.writer(csv_file)
    writerB.writerow(['game_id', 'rating'])
    for row in resultB:
        writerB.writerow(row)
    csv_file.close()
    print('Submission B complete!')

    return modelB

################# Sub C -- (non-parametric model) ###########################
## Anjali

def my_KNN(x_bot_train, x_testIDs, y_bot_train):
    # Train the model using the training sets
    modelC = KNeighborsRegressor(n_neighbors=8)
    modelC.fit(x_data, y_raw)
    print('Model C created')
    predictionC = modelC.predict(x_test)

    # rmse_train_knn=mean_squared_error(y_bot_train, predictionC,squared=False)
    # print("RMSE of Decision tree is:",rmse_train_knn)

    # Create a submission
    resultC = list(zip(x_testIDs, predictionC))

    # Output a submission
    df = pd.DataFrame(resultC)
    df.columns = ["game_id","rating"]
    filename = 'Team2_submissionC.csv'
    df.to_csv(filename, index=False)

    print('Submission C complete!')

    return modelC

################# Sub D -- (parametric model)  ###########################
## Kevin

# Parametric
def my_LinearRegression(x_data, x_test, x_testIDs, y_raw):
    #lin reg
    X_train = x_data
    y_train = np.array(y_raw)

    regr = LinearRegression()
    print('Model D built')
    regr.fit(X_train, y_train)
    y_pred = regr.predict(x_test)

    # Create a submission
    resultD = list(zip(x_testIDs, y_pred))

    # Output a submission
    df = pd.DataFrame(resultD)
    df.columns = ["game_id","rating"]
    filename = 'Team2_submissionD.csv'
    df.to_csv(filename, index=False)

    print('Submission D complete!')

    return regr

################# Sub E --(stacking model using lin reg)  ###########################
## Jason

def stacking_model(model_A, model_B, model_C, model_D, x_data, x_test, x_testIDs, y_raw):
    # define the base model
    level0 = list()
    level0.append(('NeuralNetwork', model_A))
    level0.append(('DecisionTree', model_B))
    level0.append(('KNN', model_C))
    level0.append(('LinearRegression', model_D))
    # define meta learner model
    level1 = LinearRegression()
    # define the stacking ensemble
    model_E = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
    print('Model E built')
    # fit the model on all available data
    model_E.fit(x_data, y_raw)
    # make a prediction for one example
    y_pred = model_E.predict(x_test)

    #  Create a submission
    resultE = list(zip(x_testIDs, y_pred))

    # Output a submission
    df = pd.DataFrame(resultE)
    df.columns = ["game_id","rating"]
    filename = 'Team2_submissionE.csv'
    df.to_csv(filename, index=False)

    print('Submission E complete!')

######### Main ##########

model_A = my_MLPRegressor(x_train, y_train, x_test, x_test_IDs)
model_B = my_DecisionTreeRegressor(x_train, y_train, x_test, x_test_IDs)
#model_C = my_KNN(x_bot_train, x_testIDs, y_bot_train)
#model_D = my_LinearRegression(x_train, y_train, x_test, x_test_IDs)

#stacking_model(model_A, model_B, model_C, model_D, x_data, x_test, x_testIDs, y_raw)

################# Sub D -- (parametric model)  ###########################
## Jason

def my_LogisticRegression(x_data, y_raw, x_test, x_testIDs):
    #lin reg
    X_train = x_data
    y_train = np.array(y_raw)

    regr = LogisticRegression()
    print('Model F built')
    regr.fit(X_train, y_train)
    y_pred = regr.predict(x_test)

    # Create a submission
    resultD = list(zip(x_testIDs, y_pred))

    # Output a submission
    df = pd.DataFrame(resultD)
    df.columns = ["game_id","rating"]
    filename = 'Team2_submissionF.csv'
    df.to_csv(filename, index=False)

    print('Submission F complete!')

    return regr

model_F = my_LogisticRegression(x_train, y_train, x_test, x_test_IDs)