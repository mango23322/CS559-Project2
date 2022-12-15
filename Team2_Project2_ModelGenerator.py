import csv
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Sub A -- MLP Regressor
# Sub B -- Decision Tree
# Sub C -- KNN
# Sub D -- Linear Regression
# Sub E --(stacking model using lin reg)


games = pd.read_csv('games.csv')
turns = pd.read_csv('turns.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

################# LOAD TRAIN DATA ###########################
        
#x_train_raw = train[['game_id', 'score']]
#train_raw = pd.merge(pd.merge(train, turns, on='game_id'), games, on='game_id')
train_raw = pd.merge(train, games, on='game_id')
#train_raw = train_raw.drop(columns=['nickname_x', 'nickname_y', 'rack', 'location', 'move', 'first', 'created_at'])
train_raw = train_raw.drop(columns=['nickname', 'first', 'created_at'])
#train_raw = pd.get_dummies(train_raw, columns=['turn_type', 'time_control_name', 'game_end_reason', 'lexicon', 'rating_mode'])
features_to_factorize = ['time_control_name', 'game_end_reason', 'lexicon', 'rating_mode']
for feature in features_to_factorize:
    labels, unique = pd.factorize(train_raw[feature], sort=True)
    train_raw[feature] = pd.DataFrame(data = labels, columns=[feature])
x_train_raw = train_raw.drop(columns=['rating'])
y_train_raw = train_raw[['rating']]
x_train = x_train_raw
y_train = y_train_raw
print('Training data ready')

################# LOAD TEST DATA ###########################

#x_test_raw = test[test['rating'].isnull()].loc[:, ['game_id', 'score']]
#test_raw = pd.merge(pd.merge(test, turns, on='game_id'), games, on='game_id')
test_raw = pd.merge(test, games, on='game_id')
#test_raw = test_raw.drop(columns=['nickname_x', 'nickname_y', 'rack', 'location', 'move', 'first', 'created_at'])
test_raw = test_raw.drop(columns=['nickname', 'first', 'created_at'])
#test_raw = pd.get_dummies(test_raw, columns=['turn_type', 'time_control_name', 'game_end_reason', 'lexicon', 'rating_mode'])

# Add dummy row at the end
test_raw = pd.concat([test_raw, test_raw.tail(1)], ignore_index=True)
test_raw.at[test_raw.shape[0]-1, 'lexicon'] = 'NSWL20'

#features_to_factorize = ['turn_type', 'time_control_name', 'game_end_reason', 'lexicon', 'rating_mode']
features_to_factorize = ['time_control_name', 'game_end_reason', 'lexicon', 'rating_mode']
for feature in features_to_factorize:
    labels, unique = pd.factorize(test_raw[feature], sort=True)
    test_raw[feature] = pd.DataFrame(data = labels, columns=[feature])

# Remove dummy row
test_raw = test_raw[:-1]

#df_0 = np.zeros(test_raw.shape[0])
#test_raw.insert(28, 'lexicon_NSWL20', df_0)

x_test_raw = test_raw[test_raw['rating'].isnull()].drop(columns=['rating'])
x_test_IDs_raw = test_raw[test_raw['rating'].isnull()].loc[:, ['game_id']]
x_bot_test_raw = test_raw[test_raw['rating'].notnull()].drop(columns=['rating'])
y_bot_test_raw = test_raw[test_raw['rating'].notnull()].loc[:, ['rating']]

x_test = x_test_raw
x_test_IDs = x_test_IDs_raw
x_bot_test = x_bot_test_raw
y_bot_test = y_bot_test_raw
print('Test data ready')

################# VISUALIZATION ###########################

#x = x_train[:,0]
# = x_train[:,1]
#z = y_train

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(x,y,z)
#plt.show()

################# PREPROCESSING ###########################

scalar = StandardScaler()
sc = scalar.fit(x_train)

x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
x_bot_test = sc.transform(x_bot_test)


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

def to_nparray(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test):
    return np.array(x_train), np.array(y_train).flatten(), np.array(x_test), np.array(x_test_IDs), np.array(x_bot_test), np.array(y_bot_test).flatten()

################# Sub A -- MLP Regressor ###########################

# Parametric model
def my_MLPRegressor(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test):
    x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test = to_nparray(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test)

    print('------------------------------')
    # This one takes 20 or 30 seconds on my PC
    #modelA = MLPRegressor(random_state=1, max_iter=500).fit(x_train, y_train)
    modelA = MLPRegressor()
    modelA.fit(x_train, y_train)
    print('Model A built -- NeuralNetwork')
    predictionA = modelA.predict(x_test)

    y_bot_pred = modelA.predict(x_bot_test)
    print("RMSE of MLPRegressor:", mean_squared_error(y_bot_test, y_bot_pred, squared=False))
    
    # Create a submission out of the list of game IDs and the predictions for them
    # Turn the numpy arrays back to dataframes
    x_test_IDs = pd.DataFrame(data=x_test_IDs, columns=['game_id'])
    predictionA = pd.DataFrame(data=predictionA, columns=['rating'])
    output = pd.concat([x_test_IDs, predictionA], axis=1)

    # Output a submission        
    filename = 'Team2_submissionA.csv'
    output.to_csv(filename, index=None, header=True)
    print('Submission A complete!')

    return modelA

# ################# Sub B -- Decision Tree ###########################

# Non-parametric model
def my_DecisionTreeRegressor(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test):
    x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test = to_nparray(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test)

    # Create an instance of the model
    print('------------------------------')
    modelB = DecisionTreeRegressor()
    modelB.fit(x_train, y_train)
    print('Model B built -- DecisionTree')
    predictionB = modelB.predict(x_test)

    y_bot_pred = modelB.predict(x_bot_test)
    print("RMSE of Decision tree:", mean_squared_error(y_bot_test, y_bot_pred, squared=False))

    # Create a submission
    # Turn the numpy arrays back to dataframes
    x_test_IDs = pd.DataFrame(data=x_test_IDs, columns=['game_id'])
    predictionB = pd.DataFrame(data=predictionB, columns=['rating'])
    output = pd.concat([x_test_IDs, predictionB], axis=1)

    # Output a submission
    filename = 'Team2_submissionB.csv'
    output.to_csv(filename, index=None, header=True)

    print('Submission B complete!')

    return modelB

################# Sub C -- (non-parametric model) ###########################
## Anjali

def my_KNN(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test, neighbors):
    x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test = to_nparray(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test)

    print('------------------------------')
    # Train the model using the training sets
    modelC = KNeighborsRegressor(n_neighbors=neighbors)
    modelC.fit(x_train, y_train)
    print('Model C created -- KNN')
    predictionC = modelC.predict(x_test)

    y_bot_pred = modelC.predict(x_bot_test)
    rmse = mean_squared_error(y_bot_test, y_bot_pred, squared=False)
    print("RMSE of Decision Tree:", rmse)

    # Create a submission
    # Turn the numpy arrays back to dataframes
    x_test_IDs = pd.DataFrame(data=x_test_IDs, columns=['game_id'])
    predictionC = pd.DataFrame(data=predictionC, columns=['rating'])
    output = pd.concat([x_test_IDs, predictionC], axis=1)

    # Output a submission
    filename = 'Team2_submissionC.csv'
    output.to_csv(filename, index=None, header=True)

    print('Submission C complete!')

    return modelC, rmse

#rmse_list = []
#for i in range(100, 200):
#    model, rmse = my_KNN(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test, i)
#    rmse_list.append(rmse)
#print(min(rmse_list), rmse_list.index(min(rmse_list))+1)

################# Sub D -- (parametric model)  ###########################
## Kevin

# Parametric
def my_LinearRegression(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test):
    x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test = to_nparray(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test)

    print('------------------------------')
    modelD = LinearRegression()
    print('Model D built -- LinearRegression')
    modelD.fit(x_train, y_train)
    predictionD = modelD.predict(x_test)

    #Check the accuracy
    y_bot_pred = modelD.predict(x_bot_test)
    print("RMSE of LinearRegression:", mean_squared_error(y_bot_pred, y_bot_test, squared=False))

    # Create a submission
    # Turn the numpy arrays back to dataframes
    x_test_IDs = pd.DataFrame(data=x_test_IDs, columns=['game_id'])
    predictionD = pd.DataFrame(data=predictionD, columns=['rating'])
    output = pd.concat([x_test_IDs, predictionD], axis=1)

    # Output a submission
    filename = 'Team2_submissionD.csv'
    output.to_csv(filename, index=None, header=True)

    print('Submission D complete!')

    return modelD

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

################# Sub F -- (parametric model)  ###########################
## Jason

def my_LogisticRegression(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test):
    x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test = to_nparray(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test)

    print('------------------------------')
    modelF = LogisticRegression()
    print('Model F built -- LogisticRegression')
    modelF.fit(x_train, y_train)
    predictionF = modelF.predict(x_test)

    #Check the accuracy
    y_bot_pred = modelF.predict(x_bot_test)
    print("RMSE of LogisticRegression:", mean_squared_error(y_bot_pred, y_bot_test, squared=False))

    # Create a submission
    # Turn the numpy arrays back to dataframes
    x_test_IDs = pd.DataFrame(data=x_test_IDs, columns=['game_id'])
    predictionF = pd.DataFrame(data=predictionF, columns=['rating'])
    output = pd.concat([x_test_IDs, predictionF], axis=1)

    # Output a submission
    filename = 'Team2_submissionF.csv'
    output.to_csv(filename, index=None, header=True)

    print('Submission F complete!')

    return modelF

################# Sub G -- (parametric model)  ###########################
## Jason

def my_XGBRegressor(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test):
    x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test = to_nparray(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test)
    
    print('------------------------------')
    modelG = XGBRegressor()
    #modelG = XGBRegressor(learning_rate= 0.1, n_estimators= 100, max_depth= 7, min_child_weight= 0.5, gamma= 1, subsample= 0.7, colsample_bytree=0.8, reg_alpa= 0.7, verbose=True, n_jobs= -1)
    print('Model G built -- XGBRegressor')
    modelG.fit(x_train, y_train)
    predictionG = modelG.predict(x_test)

    #Check the accuracy
    y_bot_pred = modelG.predict(x_bot_test)
    print("RMSE of XGBRegressor:", mean_squared_error(y_bot_pred, y_bot_test, squared=False))

    # Create a submission
    # Turn the numpy arrays back to dataframes
    x_test_IDs = pd.DataFrame(data=x_test_IDs, columns=['game_id'])
    predictionG = pd.DataFrame(data=predictionG, columns=['rating'])
    output = pd.concat([x_test_IDs, predictionG], axis=1)

    # Output a submission
    filename = 'Team2_submissionG.csv'
    output.to_csv(filename, index=None, header=True)

    print('Submission G complete!')

    return modelG

################# Sub H -- (parametric model)  ###########################
## Jason

def my_LGBMRegressor(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test):
    x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test = to_nparray(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test)

    print('------------------------------')
    modelH = lgb.LGBMRegressor()
    #modelH = lgb.LGBMRegressor(bagging_fraction=0.5, boosting_type='gbdt', class_weight=None, colsample_bytree=1.0, feature_fraction=0.5, importance_type='split', learning_rate=0.01, max_depth=-1, min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0, n_estimators=500, n_jobs=-1, num_leaves=130, objective='regression', random_state=42, reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
    print('Model H built -- LGBMRegressor')
    modelH.fit(x_train, y_train)
    predictionH = modelH.predict(x_test)

    #Check the accuracy
    y_bot_pred = modelH.predict(x_bot_test)
    print("RMSE of LGBMRegressor:", mean_squared_error(y_bot_pred, y_bot_test, squared=False))

    # Create a submission
    # Turn the numpy arrays back to dataframes
    x_test_IDs = pd.DataFrame(data=x_test_IDs, columns=['game_id'])
    predictionH = pd.DataFrame(data=predictionH, columns=['rating'])
    output = pd.concat([x_test_IDs, predictionH], axis=1)

    # Output a submission
    filename = 'Team2_submissionH.csv'
    output.to_csv(filename, index=None, header=True)

    print('Submission H complete!')

    return modelH

################# Sub I -- (parametric model)  ###########################
## Jason

def my_RandomForestRegressor(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test):
    x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test = to_nparray(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test)

    print('------------------------------')
    modelI = RandomForestRegressor()
    #modelI = RandomForestRegressor(n_estimators=100, n_jobs=-1, max_depth=8, verbose=True)
    print('Model H built -- RandomForestRegressor')
    modelI.fit(x_train, y_train)
    predictionI = modelI.predict(x_test)

    #Check the accuracy
    y_bot_pred = modelI.predict(x_bot_test)
    print("RMSE of RandomForestRegressor:", mean_squared_error(y_bot_pred, y_bot_test, squared=False))

    # Create a submission
    # Turn the numpy arrays back to dataframes
    x_test_IDs = pd.DataFrame(data=x_test_IDs, columns=['game_id'])
    predictionI = pd.DataFrame(data=predictionI, columns=['rating'])
    output = pd.concat([x_test_IDs, predictionI], axis=1)

    # Output a submission
    filename = 'Team2_submissionI.csv'
    output.to_csv(filename, index=None, header=True)

    print('Submission I complete!')

    return modelI

######### Main ##########

# Current RMSE:
# Model A - 180.41200852692094
# Model B - 276.42601174643426
# Model C - 180.69857811441057
# Model D - 180.30682749826582
# Model E -
# Model F - 524.6339062101499
# Model G - 123.2122684575067
# Model H - 180.7348400831483
# Model I - 209.03536804836813

#model_A = my_MLPRegressor(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test)
#model_B = my_DecisionTreeRegressor(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test)
# Best value was 700 with an RMSE of 180.69857811441057. MOst k values give an RMSE around that value.
#model_C, rmse = my_KNN(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test, 700)
#model_D = my_LinearRegression(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test)

#model_E = stacking_model(model_A, model_B, model_C, model_D, x_data, x_test, x_testIDs, y_raw)
#model_F = my_LogisticRegression(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test)
model_G = my_XGBRegressor(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test)
#model_H = my_LGBMRegressor(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test)
#model_I = my_RandomForestRegressor(x_train, y_train, x_test, x_test_IDs, x_bot_test, y_bot_test)