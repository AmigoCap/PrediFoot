import matplotlib.pyplot as plt
import numpy as np
import random

from modele_poisson import *
from modele_random import *
from elo import *

random.seed(42)

tab2006 = pd.read_csv('2006_2007.csv')
tab2007 = pd.read_csv('2007_2008.csv')
tab2008 = pd.read_csv('2008_2009.csv')
tab2009 = pd.read_csv('2009_2010.csv')
tab2010 = pd.read_csv('2010_2011.csv')
tab2011 = pd.read_csv('2011_2012.csv')
tab2012 = pd.read_csv('2012_2013.csv')
tab2013 = pd.read_csv('2013_2014.csv')
tab2014 = pd.read_csv('2014_2015.csv')
tab2015 = pd.read_csv('2015_2016.csv')
tab2016 = pd.read_csv('2016_2017.csv')
tab2017 = pd.read_csv('2017_2018.csv')


def test_prediction(tab_prec, tab, fonction_prediction):
    
    # On concatene les données de la saison précédente et de la saison actuelle
    tab_total = pd.concat((tab_prec,tab), sort=False, ignore_index = True)
    
    total_match_count = tab['FTHG'].shape[0]
    S = 0
    
    for i in range(total_match_count):
        # On enregistre le résultat de chaque match 
        
        if tab['FTR'][i] == 'H':
            res = tab['HomeTeam'][i]
        elif tab['FTR'][i] == 'A':
            res = tab['AwayTeam'][i]
        else :
            res = 'Egalité'
        
        # On a accès, pour le training, à tous les matchs précédents
        p = fonction_prediction(tab_total[190:380+i],tab['HomeTeam'][i],tab['AwayTeam'][i])
        
        if p == res : 
            S += 1
    
    resultat_prediction = S/(total_match_count) * 100
    
    return resultat_prediction


def plot_result(filename=None, display_elo=True, display_random=True, display_poisson=True):
    random.seed(42)
    

    x = ['2013-2014', '2014-2015', '2015-2016', '2016-2017', '2017-2018']
    
    if display_random:
        random2013 = test_prediction(tab2012, tab2013, prediction_random)
        random2014 = test_prediction(tab2013, tab2014, prediction_random)
        random2015 = test_prediction(tab2014, tab2015, prediction_random)
        random2016 = test_prediction(tab2015, tab2016, prediction_random)
        random2017 = test_prediction(tab2016, tab2017, prediction_random)
        
        y_random = [random2013, random2014, random2015, random2016, random2017]
        plt.plot(x, y_random, '-o', label='Modèle Aléatoire')
    
    if display_poisson:
        poisson2013 = test_prediction(tab2012, tab2013, proba_gagnant)
        poisson2014 = test_prediction(tab2013, tab2014, proba_gagnant)
        poisson2015 = test_prediction(tab2014, tab2015, proba_gagnant)
        poisson2016 = test_prediction(tab2015, tab2016, proba_gagnant)
        poisson2017 = test_prediction(tab2016, tab2017, proba_gagnant)
        
        y_poisson = [poisson2013, poisson2014, poisson2015, poisson2016, poisson2017]    
        plt.plot(x, y_poisson, '-o', label='Modèle Poisson')
    
    if display_elo:
        df = pd.concat((tab2007, tab2008, tab2009, tab2010, tab2011, tab2012, tab2013), sort=False, ignore_index = True)
        elo2013 = set_elo(df)[1]
        df = pd.concat((df, tab2014), sort=False, ignore_index = True)
        elo2014 = set_elo(df)[1]
        df = pd.concat((df, tab2015), sort=False, ignore_index = True)
        elo2015 = set_elo(df)[1]
        df = pd.concat((df, tab2016), sort=False, ignore_index = True)
        elo2016 = set_elo(df)[1]
        df = pd.concat((df, tab2017), sort=False, ignore_index = True)
        elo2017 = set_elo(df)[1]
            
        y_elo = [elo2013, elo2014, elo2015, elo2016, elo2017]
        plt.plot(x, y_elo, '-o', label='Modèle Elo')
    
    axes = plt.gca()
    axes.set_ylim(0,100)
    
    plt.xlabel('Saisons')
    #plt.xlabel(str(display_elo))
    plt.ylabel('Pourcentage de réussite')
    plt.title('Benchmark')
    plt.legend()
    
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    


def evolution_taille_training():
    result = []
    l = [47,95,142,190,237,285]
    for training_size in l:
        poisson2015 = test_prediction(tab2015[:training_size], tab2015[training_size:], proba_gagnant, training_size)
        result.append(poisson2015)
        
    plt.xlabel('Nombre de match de training')
    plt.ylabel('Pourcentage de réussite')
    plt.title('Evolution du pourcentage de réussite en fonction du nombre de match de training pour la saison 2015-2016')
    plt.plot(l, result, '-o')
    plt.show()
    

def analyse_data():
    df = pd.concat((tab2006, tab2007, tab2008, tab2009, tab2010, tab2011, tab2012, tab2013, tab2014, tab2015, tab2016, tab2017), sort=False, ignore_index = True)
    match_nb = df.shape[0]
    home_win_nb = df.loc[df['FTR']=='H'].shape[0]
    away_win_nb = df.loc[df['FTR']=='A'].shape[0]
    draw_nb = df.loc[df['FTR']=='D'].shape[0]
    
    print('Pourcentage de victoire à domicile : {}'.format(home_win_nb/match_nb))
    print('Pourcentage de victoire à l\'extérieur : {}'.format(away_win_nb/match_nb))
    print('Pourcentage d\'égalité : {}'.format(draw_nb/match_nb))

if __name__ == '__main__':
    plot_result('static/images/benchmark.png', True, True, False)
    #evolution_taille_training()
    #analyse_data()
