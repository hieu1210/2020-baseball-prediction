import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# matplotlib inline

import warnings
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
pd.options.display.max_columns = None
sns.set()
pd.options.mode.chained_assignment = None
from sklearn.model_selection import train_test_split # Data splitting for modeling
from sklearn.metrics import mean_absolute_error as mae # Error metric for models

import collections

from sklearn.linear_model import LogisticRegression

@st.cache
def loadRawData():
    df = pd.read_csv('mlb_full_games_full.csv')
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['Winner'] = df.apply(lambda x: x['Home_team_full'] if x['score_home'] > x['score_away'] else x['Away_team_full'], axis = 1)
    df['Loser'] = df.apply(lambda x: x['Home_team_full'] if x['score_home'] < x['score_away'] else x['Away_team_full'], axis = 1)
    return df

def getAnnualTeamData2(df, teamName, year):
    annual_data = df[df['Year'] == year]

    # number of games home and away
    gamesHome = annual_data[annual_data['Home_team_full'] == teamName] 
    gamesAway = annual_data[annual_data['Away_team_full'] == teamName]
    totalGames = gamesHome.append(gamesAway)
    numGames = len(totalGames.index)
    
    #Games Won Percentage = Games Won / (Games Won + Games Lost) 
    gamesWon = annual_data[annual_data['Winner'] == teamName] 
    gamesLost = annual_data[annual_data['Loser'] == teamName] 
    numGamesWon = len(gamesWon.index)
    numGamesLost = len(gamesLost.index)
    if numGames != 0:
        gamesWonPercentage = numGamesWon / numGames

    # runs per game = total Hits / total games
    totalRunsScored = gamesHome['score_home'].sum()
    # avg shots per game
    totalRunsScored += gamesAway['score_away'].sum()
    if numGames != 0:
        RunsPerGame = totalRunsScored / numGames
    # avg shots allowed per game
    totalRunsAllowed = gamesHome['score_away'].sum()
    totalRunsAllowed += gamesAway['score_home'].sum()
    if numGames != 0:
        RunsAllowedPerGame = totalRunsAllowed / numGames 

    #total AtBats
    totalAtBats = gamesHome['AB_home'].sum()
    totalAtBats += gamesAway['AB_away'].sum()

    #total Hits
    totalHits = gamesHome['H_home'].sum()
    totalHits += gamesAway['H_away'].sum()

    #total 2B (doubles)
    total2B = gamesHome['2B_home'].sum()
    total2B += gamesAway['2B_away'].sum()
    
    #total 3B (triples)
    total3B = gamesHome['3B_home'].sum()
    total3B += gamesAway['3B_away'].sum()

    # total HR (home runs)
    totalHR = gamesHome['HR_home'].sum()
    totalHR += gamesAway['HR_away'].sum()

    # total RBI (run-batted in)
    totalRBI = gamesHome['RBI_home'].sum()
    totalRBI += gamesAway['RBI_away'].sum()

    # total SF (sac flies)
    totalSF = gamesHome['SF_home'].sum()
    totalSF += gamesAway['SF_away'].sum()

    # total HBP (hit by pitches)
    totalHBP = gamesHome['HBP_home'].sum()
    totalHBP += gamesAway['HBP_away'].sum()    
    
    # total BB
    totalBB = gamesHome['BB_home'].sum()
    totalBB += gamesAway['BB_away'].sum()

    # total k
    totalk = gamesHome['k_home'].sum()
    totalk += gamesAway['k_away'].sum()

    # total teamER
    totalteamER = gamesHome['teamER_home'].sum()
    totalteamER += gamesAway['teamER_away'].sum()

    # total putouts
    totalputouts = gamesHome['putouts_home'].sum()
    totalputouts += gamesAway['putouts_away'].sum()

    #Offense stats
    OBP = (totalHits + totalBB + totalHBP) / (totalAtBats + totalBB + totalHBP + totalSF)
    total1B = totalHits - total2B - total3B - totalHR
    SLG = (1*total1B + 2*total2B + 3*total3B + 4*totalHR) / (totalAtBats)
    
    #Defense stats
    ERA = (totalteamER / (totalputouts / 3)) * 9
    
    result_dict = {'Team': [teamName], 'Game Won Percentage': [round(gamesWonPercentage, 3)],
                    'Total Runs Scored': [int(totalRunsScored)], 'Total Runs Allowed': [int(totalRunsAllowed)],
                    'Total At Bats': [int(totalAtBats)], 'Total Hits': [int(totalHits)], 'Total 2B': [int(total2B)],
                    'Total 3B': [int(total3B)], 'Total HR': [int(totalHR)], 'Total RBI': [int(totalRBI)], 
                    'Total BB': [int(totalBB)], 'Total k': [int(totalk)], 'OBP': [round(OBP, 3)], 'SLG': [round(SLG, 3)],
                    'ERA': [round(ERA, 2)]}
    result_team = pd.DataFrame.from_dict(result_dict)
    return result_team

def createAnnualDict2(df, year):
    annualDictionary = collections.defaultdict(list)
    teamList = df['Away_team_full'].unique()
    column_names = ['Team', 'Game Won Percentage',
                    'Total Runs Scored', 'Total Runs Allowed',
                    'Total At Bats', 'Total Hits', 'Total 2B',
                    'Total 3B', 'Total HR', 'Total RBI', 
                    'Total BB', 'Total k', 'OBP', 'SLG',
                    'ERA']

    result_df = pd.DataFrame(columns = column_names)

    for team in teamList:
        result_vector = getAnnualTeamData2(df, team, year)
        result_df = pd.concat([result_df, result_vector])
        # annualDictionary[team] = team_vector
    return result_df

# def getTrainingData2(df, years):
#     totalNumGames = 0
#     for year in years:
#         annual = df[df['Year'] == year]
#         totalNumGames += len(annual.index)
#     numFeatures = len(getAnnualTeamData2(df, 'Washington Nationals', 2019)) #random team, to find dimensionality
#     X = np.zeros(( totalNumGames, numFeatures))
#     y = np.zeros(( totalNumGames ))
#     indexCounter = 0
#     for year in years:
#         team_vectors = createAnnualDict2(df, year)
#         annual = df[df['Year'] == year]
#         numGamesInYear = len(annual.index)
#         xTrainAnnual = np.zeros(( numGamesInYear, numFeatures))
#         yTrainAnnual = np.zeros(( numGamesInYear ))
#         counter = 0
#         for index, row in annual.iterrows():
#             h_team = row['Home_team_full']
#             a_team = row['Away_team_full']
#             h_vector = team_vectors[h_team]
#             a_vector = team_vectors[a_team]
#             diff = [a - b for a, b in zip(h_vector, a_vector)]
#             if (counter % 2 == 0):
#                 if len(diff) != 0:
#                     xTrainAnnual[counter] = diff
#                 yTrainAnnual[counter] = 1
#             # the opposite of the difference of the vectors should be 
#             # a true negative, where team 1 does not win
#             else:
#                 if len(diff) != 0:
#                     xTrainAnnual[counter] = [ -p for p in diff]
#                 yTrainAnnual[counter] = 0
#             counter += 1
#         X[indexCounter:numGamesInYear+indexCounter] = xTrainAnnual
#         y[indexCounter:numGamesInYear+indexCounter] = yTrainAnnual
#         indexCounter += numGamesInYear
#     return X, y

# Basic preprocessing required for all the models.  
# def predictTrainData(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
#     lm = LogisticRegression()
#     lm.fit(X_train, y_train)
#     y_pred = lm.predict(X_test)
#     score = metrics.accuracy_score(y_test, y_pred) * 100
#     report = classification_report(y_test, y_pred)
#     return score, report, lm

import pickle
with open('LogisticRegression.pkl', 'rb') as model:
    reload_model = pickle.load(model)

# predictions = reload_model.predict(X_test)
# print(f'Accuracy score: {accuracy_score(y_test, predictions)}')

def createGamePrediction(new_team1_vector, new_team2_vector):
    # X_train, X_test, y_train, y_test = train_test_split(X, y)
    # lm = LogisticRegression()
    # model2 = lm.fit(X_train, y_train)
    diff = [a - b for a, b in zip(new_team1_vector, new_team2_vector)]
    diff = np.array(diff)
    diff = np.reshape(diff, (1, -1))
    predict_team = reload_model.predict(diff)
    predictions = reload_model.predict_proba(diff)
    return predict_team, predictions

def formulatePredictions(df, test_data):
    probs = [[0 for x in range(4)] for x in range(len(test_data.index))]
    for index, row in test_data.iterrows():
        year = row['Year']
        team1_Name = row['Away_team_full']
        team2_Name = row['Home_team_full']
        team1_vector = getAnnualTeamData2(df, team1_Name, year)
        team2_vector = getAnnualTeamData2(df, team2_Name, year)
        new_team1_vector = team1_vector[['Game Won Percentage', 'Total Runs Scored', 'Total Runs Allowed',
        'Total At Bats', 'Total Hits', 'Total 2B', 'Total 3B', 'Total HR', 'Total RBI', 'Total BB', 'Total k', 'OBP', 'SLG', 'ERA']]
        new_team2_vector = team2_vector[['Game Won Percentage', 'Total Runs Scored', 'Total Runs Allowed',
        'Total At Bats', 'Total Hits', 'Total 2B', 'Total 3B', 'Total HR', 'Total RBI', 'Total BB', 'Total k', 'OBP', 'SLG', 'ERA']]
        prediction = createGamePrediction(new_team1_vector.to_numpy(), new_team2_vector.to_numpy())
        predict_team = prediction[0]
        winner = None
        if predict_team == 0:
            winner = team1_Name
            loser = team2_Name
        elif predict_team == 1:
            winner = team2_Name
            loser = team1_Name
        probs[index][0] = prediction[0]
        probs[index][1] = prediction[1]
        probs[index][2] = winner
        probs[index][3] = loser
    probs = pd.np.array(probs)
    return probs

def formulatePredictions2019(df, test_2020_rev):
    probs = [[0 for x in range(4)] for x in range(len(test_2020_rev.index))]
    for index, row in test_2020_rev.iterrows():
        year = row['Year'] - 1
        team1_Name = row['Away_team_full']
        team2_Name = row['Home_team_full']
        team1_vector = getAnnualTeamData2(df, team1_Name, year)
        team2_vector = getAnnualTeamData2(df, team2_Name, year)
        new_team1_vector = team1_vector[['Game Won Percentage', 'Total Runs Scored', 'Total Runs Allowed',
        'Total At Bats', 'Total Hits', 'Total 2B', 'Total 3B', 'Total HR', 'Total RBI', 'Total BB', 'Total k', 'OBP', 'SLG', 'ERA']]
        new_team2_vector = team2_vector[['Game Won Percentage', 'Total Runs Scored', 'Total Runs Allowed',
        'Total At Bats', 'Total Hits', 'Total 2B', 'Total 3B', 'Total HR', 'Total RBI', 'Total BB', 'Total k', 'OBP', 'SLG', 'ERA']]        
        prediction = createGamePrediction(new_team1_vector.to_numpy(), new_team2_vector.to_numpy())
        predict_team = prediction[0]
        winner = None
        if predict_team == 0:
            winner = team1_Name
            loser = team2_Name
        elif predict_team == 1:
            winner = team2_Name
            loser = team1_Name
        probs[index][0] = prediction[0]
        probs[index][1] = prediction[1]
        probs[index][2] = winner
        probs[index][3] = loser
    probs = pd.np.array(probs)
    return probs

team_after2013 = {
  "Arizona D'Backs": {"conference": 'NL West', "league": 'NL'},
  "Atlanta Braves": {"conference": 'NL East', "league": 'NL'},
  "Baltimore Orioles": {"conference": 'AL East', "league": 'AL'},
  "Boston Red Sox": {"conference": 'AL East', "league": 'AL'},
  "Chicago Cubs": {"conference": 'NL Central', "league": 'NL'},
  "Chicago White Sox": {"conference": 'AL Central', "league": 'AL'},
  "Cincinnati Reds": {"conference": 'NL Central', "league": 'NL'},
  "Cleveland Indians": {"conference": 'AL Central', "league": 'AL'},
  "Colorado Rockies": {"conference": 'NL West', "league": 'NL'},
  "Detroit Tigers": {"conference": 'AL Central', "league": 'AL'},
  "Houston Astros": {"conference": 'AL West', "league": 'AL'},
  "Kansas City Royals": {"conference": 'AL Central', "league": 'AL'},
  "Los Angeles Angels": {"conference": 'AL West', "league": 'AL'},
  "Los Angeles Dodgers": {"conference": 'NL West', "league": 'NL'},
  "Miami Marlins": {"conference": 'NL East', "league": 'NL'},
  "Milwaukee Brewers": {"conference": 'NL Central', "league": 'NL'},
  "Minnesota Twins": {"conference": 'AL Central', "league": 'AL'},
  "New York Mets": {"conference": 'NL East', "league": 'NL'},
  "New York Yankees": {"conference": 'AL East', "league": 'AL'},
  "Oakland Athletics": {"conference": 'AL West', "league": 'AL'},
  "Pittsburgh Pirates": {"conference": 'NL Central', "league": 'NL'},
  "Philadelphia Phillies": {"conference": 'NL East', "league": 'NL'},
  "San Diego Padres": {"conference": 'NL West', "league": 'NL'},
  "San Francisco Giants": {"conference": 'NL West', "league": 'NL'},
  "Seattle Mariners": {"conference": 'AL West', "league": 'AL'},
  "St. Louis Cardinals": {"conference": 'NL Central', "league": 'NL'},
  "Tampa Bay Rays": {"conference": 'AL East', "league": 'AL'},
  "Texas Rangers": {"conference": 'AL West', "league": 'AL'},
  "Toronto Blue Jays": {"conference": 'AL East', "league": 'AL'},
  "Washington Nationals": {"conference": 'NL East', "league": 'NL'}
}

def get_info(x, mode):
    return team_after2013[str(x)][mode]

# def accept_user_data():
# 	date = st.date_input("Enter the date: ")
# 	away_team = st.text_input("Enter the away team: ")
# 	home_team = st.text_input("Enter the home team: ")
# 	user_prediction_data = np.array([date,away_team,home_team]).reshape(1,-1)

# 	return user_prediction_data

def formulatePredictions2018(df, test_2021):
    probs = [[0 for x in range(4)] for x in range(len(test_2021.index))]
    for index, row in test_2021.iterrows():
        year = row['Year'] - 3
        team1_Name = row['Away_team_full']
        team2_Name = row['Home_team_full']
        team1_vector = getAnnualTeamData2(df, team1_Name, year)
        team2_vector = getAnnualTeamData2(df, team2_Name, year)
        new_team1_vector = team1_vector[['Game Won Percentage', 'Total Runs Scored', 'Total Runs Allowed',
        'Total At Bats', 'Total Hits', 'Total 2B', 'Total 3B', 'Total HR', 'Total RBI', 'Total BB', 'Total k', 'OBP', 'SLG', 'ERA']]
        new_team2_vector = team2_vector[['Game Won Percentage', 'Total Runs Scored', 'Total Runs Allowed',
        'Total At Bats', 'Total Hits', 'Total 2B', 'Total 3B', 'Total HR', 'Total RBI', 'Total BB', 'Total k', 'OBP', 'SLG', 'ERA']]   
        prediction = createGamePrediction(new_team1_vector.to_numpy(), new_team2_vector.to_numpy())
        predict_team = prediction[0]
        winner = None
        if predict_team == 0:
            winner = team1_Name
            loser = team2_Name
        elif predict_team == 1:
            winner = team2_Name
            loser = team1_Name
        probs[index][0] = prediction[0]
        probs[index][1] = prediction[1]
        probs[index][2] = winner
        probs[index][3] = loser
    probs = pd.np.array(probs)
    return probs

def formulatePredictions2017(df, test_2020_org):
    probs = [[0 for x in range(4)] for x in range(len(test_2020_org.index))]
    for index, row in test_2020_org.iterrows():
        year = row['Year'] - 3
        team1_Name = row['Away_team_full']
        team2_Name = row['Home_team_full']
        team1_vector = getAnnualTeamData2(df, team1_Name, year)
        team2_vector = getAnnualTeamData2(df, team2_Name, year)
        new_team1_vector = team1_vector[['Game Won Percentage', 'Total Runs Scored', 'Total Runs Allowed',
        'Total At Bats', 'Total Hits', 'Total 2B', 'Total 3B', 'Total HR', 'Total RBI', 'Total BB', 'Total k', 'OBP', 'SLG', 'ERA']]
        new_team2_vector = team2_vector[['Game Won Percentage', 'Total Runs Scored', 'Total Runs Allowed',
        'Total At Bats', 'Total Hits', 'Total 2B', 'Total 3B', 'Total HR', 'Total RBI', 'Total BB', 'Total k', 'OBP', 'SLG', 'ERA']]   
        prediction = createGamePrediction(new_team1_vector.to_numpy(), new_team2_vector.to_numpy())
        predict_team = prediction[0]
        winner = None
        if predict_team == 0:
            winner = team1_Name
            loser = team2_Name
        elif predict_team == 1:
            winner = team2_Name
            loser = team1_Name
        probs[index][0] = prediction[0]
        probs[index][1] = prediction[1]
        probs[index][2] = winner
        probs[index][3] = loser
    probs = pd.np.array(probs)
    return probs

def main():
    st.title("Prediction of Baseball Game Outcomes using ML Algorithm – A Streamlit Demo")
    MLB_data = loadRawData()

    from PIL import Image
    image = Image.open('0FfgdjZ.jpg')
    st.image(image, caption='Baseball', use_column_width=True)

    # years = range(2019, 2021)
    # X, y = getTrainingData2(MLB_data, years)

	# Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
        st.subheader("Showing raw data---->>>")
        number = st.number_input("Number of Rows to View", 1, 100)
        st.dataframe(MLB_data.tail(number))

    #Show Columns
    if st.button("Column Names"):
        st.write(MLB_data.columns)

    #Show Shape
    if st.checkbox("Shape of Dataset"):
        st.write(MLB_data.shape)
        data_dim = st.radio("Show Dimensions By ", ("Rows", "Columns"))
        if data_dim == 'Row':
            st.text("Number of Rows")
            st.write(MLB_data.shape[0])
        elif data_dim == 'Columns':
            st.text("Number of Columns")
            st.write(MLB_data.shape[1])
        else:
            st.write(MLB_data.shape)

    #Select Columns
    if st.checkbox("Select Columns To Show"):
        all_columns = MLB_data.columns.tolist()
        selected_columns = st.multiselect("Select", all_columns)
        new_MLB_data = MLB_data[selected_columns]
        st.dataframe(new_MLB_data)

    #Show Values
    if st.button("Data Types"):
        st.write(MLB_data.dtypes)

    #Show Summary
    if st.checkbox("Summary"):
        st.write(MLB_data.describe().T)

    ## Plot and Visualization
    st.subheader("Data Visualization")
    # Correlation
    # Seaborn Plot
    if st.checkbox("Correlation Plot[Seaborn]"):
        st.write(sns.heatmap(MLB_data.corr(), annot=True))
        st.pyplot()

    # Pie Chart
    if st.checkbox("Pie Plot"):
        all_columns_names = MLB_data.columns.tolist()
    if st.button("Generate Plot"):
      st.success("Generating A Pie Plot")
      st.write(MLB_data.iloc[:, 1].value_counts().plot.pie(autopct="%1.1f%%"))
      st.pyplot()

    # Count Plot
    if st.checkbox("Plot of Value Counts"):
        st.text("Value Counts by Target")
        all_columns_names = MLB_data.columns.tolist()
        primary_col = st.selectbox("Primary Column to GroupBy", all_columns_names)
        selected_columns_names = st.multiselect("Select Columns", all_columns_names)
        if st.button("Plot", key='0001'):
            st.text("Generate Plot")
        if selected_columns_names:
            vc_plot = MLB_data.groupby(primary_col)[selected_columns_names].count()
        else:
            vc_plot = MLB_data.iloc[:, -1].value_counts()
        st.write(vc_plot.plot(kind="bar"))
        st.pyplot()  

    # Customizable Plot

    st.subheader("Customizable Plot")
    all_columns_names = MLB_data.columns.tolist()
    type_of_plot = st.selectbox("Select Type of Plot", ["area", "bar", "line", "hist", "box", "kde"], key='0004')
    selected_columns_names = st.multiselect("Select Columns To Plot", all_columns_names)
    
    if st.button("Generate Plot", key='0002'):
        st.success("Generating Customizable Plot of {} for {}".format(type_of_plot, selected_columns_names))

    # Plot by Streamlit
    if type_of_plot == 'area':
      cust_data = MLB_data[selected_columns_names]
      st.area_chart(cust_data)
    elif type_of_plot == 'bar':
      cust_data = MLB_data[selected_columns_names]
      st.bar_chart(cust_data)
    elif type_of_plot == 'line':
      cust_data = MLB_data[selected_columns_names]
      st.line_chart(cust_data)

    elif type_of_plot:
      cust_plot = MLB_data[selected_columns_names].plot(kind=type_of_plot)
      st.write(cust_plot)
      st.pyplot()

    if st.checkbox("Getting A Team Stat by Year"):
        teamName = st.text_input("Enter the Team: ")
        year = st.number_input("Enter the Year: ", 2013, 2020, key = "041195")
        if st.button("Get", key="0245"):
            result_df = getAnnualTeamData2(MLB_data, teamName, year)
            st.write(result_df)

    if st.checkbox("Getting All Team Stats by Year"):
        year = st.number_input("Enter the Year: ", 2013, 2020, key = "041196")
        if st.button("Get", key="0246"):
            result_df = createAnnualDict2(MLB_data, year)
            st.write(result_df)

            plot_data = result_df[['Team', 'Game Won Percentage']].sort_values(by='Game Won Percentage', ascending=True)
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.barh(plot_data['Team'].tolist(), plot_data['Game Won Percentage'].tolist())
            ax.set(title="Game Won Percentage by Team", xlabel = 'Game Won Percentage', ylabel = 'Team')
            st.pyplot(fig)

            plot_data = result_df[['Team', 'ERA']].sort_values(by='ERA', ascending=False)
            fig1, ax = plt.subplots(figsize=(10, 10))
            ax.barh(plot_data['Team'].tolist(), plot_data['ERA'].tolist())
            ax.set(title="ERA by Team", xlabel = 'ERA', ylabel = 'Team')
            st.pyplot(fig1)
            
            plot_data = result_df[['Team', 'Total Runs Scored', 'Total Runs Allowed']]
            x = np.arange(30)
            w = 0.4
            fig2, ax = plt.subplots(figsize=(10, 10))
            plt.xticks(x+w/2, plot_data['Team'], rotation='vertical')
            runs_scored = ax.bar(x, plot_data['Total Runs Scored'].tolist(), width=w, color='b', align='center')
            ax1 = ax.twinx()
            runs_allowed = ax1.bar(x+w, plot_data['Total Runs Allowed'].tolist(), width=w, color='r', align='center')
            plt.legend([runs_scored, runs_allowed], ['Runs Scored', 'Runs Allowed'])
            # plt.ylabel('Runs')
            ax.set(title="Runs per Team", xlabel = 'Team', ylabel = 'Runs')
            st.pyplot(fig2)

    # if st.checkbox("Getting Training Data"):
    #     begin_year = st.number_input("Enter the Year: ", 2013, 2020)
    #     end_year = st.number_input("Enter the Year: ", 2014, 2021)
    #     if st.button("Predict"):
    #         years = range(begin_year, end_year)
    #         X, y = getTrainingData2(MLB_data, years)
    #         score, report, lm = predictTrainData(X, y)
    #         st.text("Accuracy of Training Data model is: ")
    #         st.write(score,"%")
    #         st.text("Report of Training Data model is: ")
    #         st.write(report)
            
    choose_testing = st.sidebar.selectbox("Choose the Testing Method", ["None", "Up-to-date 2020 Season", "Full Revised 2020 Season", "Full Original 2021 Season", "Full Original 2020 Season"])

    if (choose_testing == "Up-to-date 2020 Season"):
        test_full = pd.read_csv('Test.csv')
        test_full['Date'] = pd.to_datetime(test_full['Date'], utc = False)
        outcome = formulatePredictions(MLB_data, test_full)
        draft = test_full.iloc[:, np.r_[5, 10, 12, 15, 17]]
        draft['winning_team'] = outcome[:, 2]
        st.write(draft)
        draft['losing_team'] = outcome[:, 3]
        winning = pd.DataFrame(draft.groupby('winning_team')['winning_team'].count())
        losing = pd.DataFrame(draft.groupby('losing_team')['losing_team'].count())
        tally = pd.concat([winning, losing], axis=1)
        tally.reset_index(inplace=True)
        tally = tally.rename(columns={"index": "team", "winning_team": "W", "losing_team": "L"})
        tally['conference'] = tally['team'].map(lambda x: get_info(x, 'conference'))
        tally['league'] = tally['team'].map(lambda x: get_info(x, 'league'))
        tally = tally[['team', 'conference', 'league', 'W', 'L']]
        tally = tally.sort_values(by=['conference', 'W'], ascending=[True, False])
        st.write(tally)
            # try:
            #     if(st.checkbox("Want to predict on your own Input?")):
            #         user_prediction_data = accept_user_data() 		
            #         pred = tree.predict(user_prediction_data)
            #         st.write("The Predicted Class is: ", le.inverse_transform(pred)) # Inverse transform to get the original dependent value. 
            # except:
            #     pass

    if (choose_testing == 'Full Revised 2020 Season'):
        test_2020_rev = pd.read_csv('2020_rev_test.csv')
        test_2020_rev['Date'] = pd.to_datetime(test_2020_rev['Date'], utc = False)
        outcome = formulatePredictions2019(MLB_data, test_2020_rev)
        draft = test_2020_rev.iloc[:, np.r_[1, 6:9, 11:14]]
        draft['winning_team'] = outcome[:, 2]
        draft['losing_team'] = outcome[:, 3]
        winning = pd.DataFrame(draft.groupby('winning_team')['winning_team'].count())
        losing = pd.DataFrame(draft.groupby('losing_team')['losing_team'].count())
        tally = pd.concat([winning, losing], axis=1)
        tally.reset_index(inplace=True)
        tally = tally.rename(columns={"index": "team", "winning_team": "W", "losing_team": "L"})
        tally['conference'] = tally['team'].map(lambda x: get_info(x, 'conference'))
        tally['league'] = tally['team'].map(lambda x: get_info(x, 'league'))
        tally = tally[['team', 'conference', 'league', 'W', 'L']]
        tally = tally.sort_values(by=['conference', 'W'], ascending=[True, False])
        st.write(tally)
            
    if (choose_testing == 'Full Original 2021 Season'):
        test_2021 = pd.read_csv('2021SKED.csv')
        test_2021['Date'] = pd.to_datetime(test_2021['Date'], utc = False)
        outcome = formulatePredictions2018(MLB_data, test_2021)
        draft = test_2021.iloc[:, np.r_[1, 6:9, 11:14]]
        draft['winning_team'] = outcome[:, 2]
        draft['losing_team'] = outcome[:, 3]
        winning = pd.DataFrame(draft.groupby(['winning_team'])['winning_team'].count())
        losing = pd.DataFrame(draft.groupby(['losing_team'])['losing_team'].count())
        tally = pd.concat([winning, losing], axis=1)
        tally.reset_index(inplace=True)
        tally = tally.rename(columns={"index": "team", "winning_team": "W", "losing_team": "L"})
        tally['conference'] = tally['team'].map(lambda x: get_info(x, 'conference'))
        tally['league'] = tally['team'].map(lambda x: get_info(x, 'league'))
        tally = tally[['team', 'conference', 'league', 'W', 'L']]
        tally = tally.sort_values(by=['conference', 'W'], ascending=[True, False])
        st.write(tally)
            
    if (choose_testing == 'Full Original 2020 Season'):
        test_2020_org = pd.read_csv('2020_org_test.csv')
        test_2020_org['Date'] = pd.to_datetime(test_2020_org['Date'], utc = False)
        outcome = formulatePredictions2017(MLB_data, test_2020_org)
        draft = test_2020_org.iloc[:, np.r_[1, 6:9, 11:14]]
        draft['winning_team'] = outcome[:, 2]
        draft['losing_team'] = outcome[:, 3]
        winning = pd.DataFrame(draft.groupby(['winning_team'])['winning_team'].count())
        losing = pd.DataFrame(draft.groupby(['losing_team'])['losing_team'].count())
        tally = pd.concat([winning, losing], axis=1)
        tally.reset_index(inplace=True)
        tally = tally.rename(columns={"index": "team", "winning_team": "W", "losing_team": "L"})
        tally['conference'] = tally['team'].map(lambda x: get_info(x, 'conference'))
        tally['league'] = tally['team'].map(lambda x: get_info(x, 'league'))
        tally = tally[['team', 'conference', 'league', 'W', 'L']]
        tally = tally.sort_values(by=['conference', 'W'], ascending=[True, False])
        st.write(tally)

    if st.button("Thanks"):
        st.balloons()

if __name__ == "__main__":
	main()