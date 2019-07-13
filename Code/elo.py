import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import matplotlib.dates as mdates
from math import ceil, floor

""" Implement an elo model based on this article : 
    - https://stuartlacy.co.uk/2017/08/31/implementing-an-elo-rating-system-for-european-football/
    
    Uses this article to convert elo difference into winning probability :
    - https://betting.betfair.com/football/premier-league/barclays-premier-league-coverting-ratings-to-odds-030515-186.html
"""

def expected_outcome(dr):
    """ dr is the elo difference between both teams
    dr = elo_home - elo_away """
    
    return 1 / (1 + 10**(-dr/400))

def get_G(MOV, dr):
    if MOV <= 1:
        return 1
    else :
        # Add a multiplicative factor for it to fit the curb we want
        factor = 1.444405820290127 
        return np.log(1.7 * MOV) * (2 / (2 + 0.001*dr)) * factor

def observed_outcome(df, index):
    """ Returns 0 if away_win, 0.5 if draw, 1 if home_win"""
    if df['FTR'][index] == 'A':
        return 0
    elif df['FTR'][index] == 'D':
        return 0.5
    elif df['FTR'][index] == 'H':
        return 1

def get_team_changes(df, s, season_size=380):
    """ Return the list of promoted teams and the list of relegated teams """
    previous_team_list = list(df[(s-1)*season_size : s*season_size]['HomeTeam'].drop_duplicates())
    next_team_list = list(df[s*season_size : (s+1)*season_size]['HomeTeam'].drop_duplicates())
    
    relegated_teams = [team for team in previous_team_list if team not in next_team_list]
    promoted_teams = [team for team in next_team_list if team not in previous_team_list]
    
    #print("Season {} : \n relegated teams : {} \n Promoted teams : {}".format(s, relegated_teams, promoted_teams))
    return relegated_teams, promoted_teams

    
def set_elo(df, match_nb=380):
    
    prediction_list = []
    
    good_prediction_nb = 0
    E_predictions = {}
    E_predictions['H'] = []
    E_predictions['D'] = []
    E_predictions['A'] = []
    
    # There are 380 match in a season
    season_size = match_nb
    season_nb = ceil(df.shape[0] / season_size)
    
    elo = {}
    
    # Initialize the elo for each teams
    first_season_teams = df[:season_size]['HomeTeam'].drop_duplicates()
    for team in first_season_teams:
        elo[team] = [(1500, df['Date'][0])]
    

    for s in range(season_nb):
        
        if s != 0:
            relegated_teams, promoted_teams = get_team_changes(df, s, season_size)
            
            # Get average elo of relegated teams
            avg_elo = 0
            for team in relegated_teams:
                avg_elo += elo[team][-1][0]
            avg_elo = avg_elo / len(relegated_teams)
            
            for team in promoted_teams:
                elo[team] = [(avg_elo, df['Date'][s*season_size]) ]
            
        for i in range(season_size):
            index = s * season_size + i
            
            if index >= len(df['HomeTeam']):
                break
            
            home_team = df['HomeTeam'][index]
            away_team = df['AwayTeam'][index]
            date = df['Date'][index]
            
            #print(date)
                
            #print(home_team)
            elo_home = elo[home_team][-1][0]
            elo_away = elo[away_team][-1][0]
                
            # Set an arbitrary home_advantage value
            home_advantage = 75
            dr = elo_home - elo_away + home_advantage
                
            E = expected_outcome(dr)
            O = observed_outcome(df, index)
            K = 20
            
            if s == season_nb - 1:
                # Get data for predictions on the last season
                success = is_prediction_correct(df, elo, index, E, E_predictions, prediction_list)
                
                if success:
                    # Prediction was correct
                    good_prediction_nb += 1
            
                
            margin_of_victory = abs(df['FTHG'][index] - df['FTAG'][index])
            G = get_G(margin_of_victory, dr)
            
            #print(index)
            #print(f"Domicile : {df['FTHG'][index]}, away : {df['FTAG'][index]}")
            #print(f"Margin of victory : {margin_of_victory}, G : {G}")
                
            new_elo_home = elo_home + K*G*(O - E)
                
            delta = new_elo_home - elo_home
            new_elo_away = elo_away - delta
                
            elo[home_team].append((new_elo_home, date))
            elo[away_team].append((new_elo_away, date))
        
        # At the end of the season, bring back each team's rating toward the mean
        
        if s != season_nb - 1:
            for team in elo:
                new_elo = 0.8*elo[team][-1][0] + 0.2*1500
                
                # Get the date one day after last season match
                date = df['Date'][s*season_size + season_size - 1]
                day_string = date[:2]
                new_date = str(int(day_string) + 1) + date[2:]
                
                elo[team].append((new_elo, new_date))
    
    return elo, (good_prediction_nb/season_size) * 100, E_predictions, prediction_list

def plot_elo(elo, team_list, filename=None, plot_length=200):
    
    fig = plt.figure()
    ax = plt.subplot(111)
    
    x = [i for i in range(plot_length)]
    
    for team in team_list:
        
        if plot_length >= len(elo[team]):
            starting_point = 0
        else :
            starting_point = len(elo[team]) - plot_length
            
        elo_team = [elo[team][i][0] for i in range(starting_point, len(elo[team]))]
        
        
        x = [plot_length - i for i in range(len(elo_team)-1, -1, -1)]

        ax.plot(x, elo_team, label=team)
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    
    plt.xlabel('Nb de matchs')
    plt.ylabel('Points Elo')
    
    ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
    
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

def plot_elo2(elo, team_list, filename=None):
    fig = plt.figure(figsize=(8, 6), dpi=80)
    ax = plt.subplot(111)    
    
    for team in team_list:
        dates_str = []
        elo_list = []
        for i in range(len(elo[team])):
            date_str = elo[team][i][1]
            elo_team = elo[team][i][0]
            
            if len(date_str[6:]) > 2 :
                year = int(date_str[6:])
            else :
                year = int('20' + date_str[6:])
            month = int(date_str[3:5])
            day = int(date_str[:2])
            date = datetime.date(year,month, day)
            
            dates_str.append(date)
            elo_list.append(elo_team)

        
        ax.plot(dates_str, elo_list, label=team)


    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    
    plt.xlabel('Date')
    plt.ylabel('Points Elo')
    
    ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
    
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    
    
    
def is_prediction_correct(df, elo, index, E, E_predictions, prediction_list = None):
    
#    E_prob = {}
#    E_prob[0] = (0,0,1)
#    E_prob[0.1] = (0.01,0.18,0.81)
#    E_prob[0.2] = (0.04,0.32,0.64)
#    E_prob[0.3] = (0.09,0.42,0.49)
#    E_prob[0.4] = (0.16,0.48,0.36)
#    E_prob[0.5] = (0.25,0.50,0.25)
#    E_prob[0.6] = (0.36,0.48,0.16)
#    E_prob[0.7] = (0.49,0.42,0.09)
#    E_prob[0.8] = (0.64,0.32,0.04)
#    E_prob[0.9] = (0.81,0.18,0.01)
#    E_prob[1] = (1,0,0)
#    
#    
#    inf = float(str(E)[:3])
#    sup = float(str(E+0.1)[:3])
#    difference = E - inf
#    
#    home_win_prob = (1-difference)*E_prob[inf][0] + difference*E_prob[sup][0]
#    draw_prob = (1-difference)*E_prob[inf][1] + difference*E_prob[sup][1]
#    away_win_prob = (1-difference)*E_prob[inf][2] + difference*E_prob[sup][2]
    


    # Statistiques sur l'ensemble des saisons depuis 2006
#    home_win_prob = 0.46228070175438596
#    away_win_prob = 0.2824561403508772
#    draw_prob = 0.25526315789473686
    
    home_team = df['HomeTeam'][index]
    away_team = df['AwayTeam'][index]
    
    elo_home = elo[home_team][-1][0]*1.07
    elo_away = elo[away_team][-1][0]
    
    if elo_home > elo_away:
        home_win_prob = 35 + floor((elo_home - elo_away)/10)
        draw_prob = 30 - floor(((elo_home - elo_away)-100)/20)
        away_win_prob = 100 - (home_win_prob + draw_prob)
    else :
        away_win_prob = 35 + floor((elo_away - elo_home)/10)
        draw_prob = 30 - floor(((elo_away - elo_home)-100)/20)
        home_win_prob = 100 - (away_win_prob + draw_prob)
    
    
    
    prob = np.array([home_win_prob/100, draw_prob/100, away_win_prob/100])
    
    divider = np.exp(prob[0]) + np.exp(prob[1]) + np.exp(prob[2])
    
    prob_normalized = np.exp(prob)/divider
    
    home_win_prob = prob_normalized[0]
    draw_prob = prob_normalized[1]
    away_win_prob = prob_normalized[2]
    
    
    
    
    prediction_list.append([home_win_prob, draw_prob, away_win_prob])
    
    

    if max(away_win_prob, draw_prob, home_win_prob) == home_win_prob:
        E_predictions['H'].append(E)
        prediction = home_team
    elif max(away_win_prob, draw_prob, home_win_prob) == draw_prob:
        E_predictions['D'].append(E)
        prediction = 'Egalité'
    else:
        E_predictions['A'].append(E)
        prediction = away_team                

    
    if df['FTR'][index] == 'H':
        result = df['HomeTeam'][index]
    elif df['FTR'][index] == 'A':
        result = df['AwayTeam'][index]
    else :
        result = 'Egalité'   
    
    return prediction == result


def get_elo_probabilities(df, elo, home_team, away_team):
    
    elo_home = elo[home_team][-1][0]*1.07
    elo_away = elo[away_team][-1][0]
    
    if elo_home > elo_away:
        home_win_prob = 35 + floor((elo_home - elo_away)/10)
        draw_prob = 30 - floor(((elo_home - elo_away)-100)/20)
        away_win_prob = 100 - (home_win_prob + draw_prob)
    else :
        away_win_prob = 35 + floor((elo_away - elo_home)/10)
        draw_prob = 30 - floor(((elo_away - elo_home)-100)/20)
        home_win_prob = 100 - (away_win_prob + draw_prob)
    
    
    prob = np.array([home_win_prob/100, draw_prob/100, away_win_prob/100])
    
    divider = np.exp(prob[0]) + np.exp(prob[1]) + np.exp(prob[2])
    
    prob_normalized = np.exp(prob)/divider
    
    home_win_prob = int(prob_normalized[0]*1000)/10
    draw_prob = int(prob_normalized[1]*1000)/10
    away_win_prob = int(prob_normalized[2]*1000)/10
    return (home_win_prob, draw_prob, away_win_prob)
    
def plot_G():
    for dr in range(0,400,100):
        x = [0,1,2,3,4,5,6,7,8]
        y = []
        for MOV in range(9):
            y.append(get_G(MOV, dr))
        plt.plot(x,y,label="dr = " + str(dr))
        plt.xlabel('MOV')
        plt.ylabel('G')
        plt.legend()
    plt.show()
            
            
            
        
        
    



    
if __name__ == '__main__':
#    tab2006 = pd.read_csv('static/csv_uk/2006_2007.csv')
#    tab2007 = pd.read_csv('static/csv_uk/2007_2008.csv')
#    tab2008 = pd.read_csv('static/csv_uk/2008_2009.csv')
#    tab2009 = pd.read_csv('static/csv_uk/2009_2010.csv')
#    tab2010 = pd.read_csv('static/csv_uk/2010_2011.csv')
#    tab2011 = pd.read_csv('static/csv_uk/2011_2012.csv')
#    tab2012 = pd.read_csv('static/csv_uk/2012_2013.csv')
#    tab2013 = pd.read_csv('static/csv_uk/2013_2014.csv')
#    tab2014 = pd.read_csv('static/csv_uk/2014_2015.csv')
#    tab2015 = pd.read_csv('static/csv_uk/2015_2016.csv')
#    tab2016 = pd.read_csv('static/csv_uk/2016_2017.csv')
#    tab2017 = pd.read_csv('static/csv_uk/2017_2018.csv')
#    tab2018 = pd.read_csv('static/csv_uk/2018_2019.csv')
    
    dir_name = 'static/csv_uk/'
    season_size = 380
    
    tab2006 = pd.read_csv(dir_name + '2006_2007.csv')
    tab2007 = pd.read_csv(dir_name + '2007_2008.csv')
    tab2008 = pd.read_csv(dir_name + '2008_2009.csv')
    tab2009 = pd.read_csv(dir_name + '2009_2010.csv')
    tab2010 = pd.read_csv(dir_name + '2010_2011.csv')
    tab2011 = pd.read_csv(dir_name + '2011_2012.csv')
    tab2012 = pd.read_csv(dir_name + '2012_2013.csv')
    tab2013 = pd.read_csv(dir_name + '2013_2014.csv')
    tab2014 = pd.read_csv(dir_name + '2014_2015.csv')
    tab2015 = pd.read_csv(dir_name + '2015_2016.csv')
    tab2016 = pd.read_csv(dir_name + '2016_2017.csv')
    tab2017 = pd.read_csv(dir_name + '2017_2018.csv')
    tab2018 = pd.read_csv(dir_name + '2018_2019.csv')
    
    
    df = pd.concat((tab2006, tab2007, tab2008, tab2009, tab2010, tab2011, tab2012, tab2013, tab2014, tab2015, tab2016, tab2017, tab2018), sort=False, ignore_index = True)
    
    elo, percentage, E_predictions, prediction_list = set_elo(df)
    
    print(f"Pourcentage de réussite : {percentage}")
    for res in E_predictions:
        print(f"Nombre de {res} : {len(E_predictions[res])} ")
        print(f"Pourcentage de {res} : {len(E_predictions[res])/season_size}")
        
    team_list = ['Man City', 'Liverpool'] #list(tab2018['HomeTeam'].drop_duplicates())


    def elo_sort(team):
        return elo[team][-1][0]
    
    team_list.sort(key=elo_sort, reverse=True)
    

    for team in team_list:
        print('Elo de {} : {}'.format(team, elo[team][-1]))

    
    plot_elo2(elo, team_list, 'static/images/elo.png')
    
    #plot_G()












#def predict_match(df, i, E_proportion):
#    
#    elo = set_elo(df, i)
#    
#    home_team = df['HomeTeam'][i]
#    away_team = df['AwayTeam'][i]
#    
#    elo_home = elo[home_team][-1]
#    elo_away = elo[away_team][-1]
#    # Set an arbitrary home_advantage value
#    home_advantage = 80
#    dr = elo_home - elo_away + home_advantage
#    
#    E = expected_outcome(dr)
#    
#    home_win_prob = 0.46228070175438596
#    away_win_prob = 0.2824561403508772
#    draw_prob = 0.25526315789473686
#    
#    if E < home_win_prob:
#        E_proportion['H'].append(E)
#        return home_team
#    elif E < home_win_prob + draw_prob:
#        E_proportion['D'].append(E)
#        return 'Egalité'
#    else :
#        E_proportion['A'].append(E)
#        return away_team
#
#def test_one_season(df):
#    """Gives percentage of success on the last season that is part of df """
#    
#    S = 0
#    start_point = df.shape[0] - 380
#    
#    E_proportion = {}
#    E_proportion['H'] = []
#    E_proportion['D'] = []
#    E_proportion['A'] = []
#
#    for i in range(start_point, df.shape[0]):
#        print(i-start_point)
#        if df['FTR'][i] == 'H':
#            res = df['HomeTeam'][i]
#        elif df['FTR'][i] == 'A':
#            res = df['AwayTeam'][i]
#        else :
#            res = 'Egalité'
#        
#        prediction = predict_match(df, i-start_point, E_proportion)
#        
#        if prediction == res:
#            S += 1
#        
#    
#    return (S/380) * 100, E_proportion
    
        
  