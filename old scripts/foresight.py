import pandas as pd
import numpy as np
import random as rnd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import scipy.stats as stats

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# machine learning
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model


def main():

    warnings.filterwarnings("ignore")
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    df = pd.read_csv('HomeC.csv', delimiter=',')
    df.rename(columns={'use [kW]': 'use', 'gen [kW]': 'gen', 'House overall [kW]': 'House overall', 'Dishwasher [kW]': 'Dishwasher', 'Furnace 1 [kW]': 'Furnace1', 'Furnace 2 [kW]': 'Furnace2', 'Home office [kW]': 'Home office', 'Fridge [kW]': 'Fridge', 'Wine cellar [kW]': 'Wine cellar', 'Garage door [kW]': 'Garage door', 'Kitchen 12 [kW]': 'Kitchen12', 'Kitchen 14 [kW]': 'Kitchen14', 'Kitchen 38 [kW]': 'Kitchen38', 'Barn [kW]': 'Barn', 'Well [kW]': 'Well', 'Microwave [kW]': 'Microwave', 'Living room [kW]': 'Living room', 'Solar [kW]': 'Solar'}, inplace=True)
    del df['gen'], df['apparentTemperature'], df['House overall'], df['dewPoint'], df['precipIntensity']
    dataset = df_preprocessing(df)

    #                    0,23 bei nicht <3 100k
    # 0,28; 0,29         0,37; 0,38 bei nicht <3 10k
    # 0,38; 0,38         0,55; 0,53 bei nicht <3 1k
    # 0,   100k
    # 0,28; 0,29         0,37; 0,38 bei nicht <2,5 10k

    Trainingsvolumen = 1000
    Testvolumen = 2000

    # Herausfiltern der Ausreißer
    dataset = dataset.loc[dataset['use'] < 2.5]
    # mit mixen, siehe auch in der prediction schleife
    #dataset = dataset.sample(frac=1).reset_index(drop=True)
    # Zusammenfassen von 60 min zu einer Zeile
    # dataset = dataset.groupby(['month', 'monthday', 'hour']).mean()
    # ohne Hausgeräte
    # del dataset['Dishwasher'], dataset['Home office'], dataset['Fridge'], dataset['Wine cellar'], dataset['Garage door'], dataset['Barn'], dataset['Well'], dataset['Microwave'], dataset['Living room'], dataset['Furnace'], dataset['Kitchen'], dataset['Solar']
    # ohne Wettereinfluss
    # del dataset['temperature'], dataset['icon'], dataset['humidity'], dataset['visibility'], dataset['summary'], dataset['pressure'], dataset['windSpeed'], dataset['cloudCover'], dataset['windBearing'], dataset['precipProbability']
    # für Weihnachtstage
    # X_testt = dataset.tail(Testvolumen)
    # für folgende Tage
    X_testt = (dataset.head(Trainingsvolumen+Testvolumen)).tail(Testvolumen)


    dataset = dataset.head(Trainingsvolumen)
    with open('Testset.textmate', 'w') as file:
        file.write(str(X_testt) + '\n')
    with open('Trainingset.textmate', 'w') as file:
        file.write(str(dataset) + '\n')
    df_statistics = df_stats(dataset)
    df_statistics0 = df_stats0(X_testt)
    with open('Infos Testset.textmate', 'w') as file:
        file.write(str(df_statistics0) + '\n')
    #corr = correlation(dataset)


    i = int(len(dataset))
    results = pd.DataFrame()
    highrate = pd.DataFrame()

    for x in range(0, 1):
        data = dataset
        results = predictions(data, i, X_testt, df_statistics)
    corr = correlation2(results)
    mse = rmse(results)
    highrate = highrated(results)
    with open('df_results.textmate', 'w') as file:
        file.write(str(results) + '\n')
    highrate.to_csv('Höchste Bewertungen 2020.csv')


def df_preprocessing(df):

    print('Abschlusswahrscheinlichkeit wird berechnet...')

    #print(df[["use", "summary"]].groupby(['summary'], as_index=False).mean().sort_values(by='use',ascending=False))
    df = df[0:-1]

    # icon
    df['icon'] = df['icon'].replace('fog', 9)
    df['icon'] = df['icon'].replace('clear-day', 8)
    df['icon'] = df['icon'].replace('clear-night', 7)
    df['icon'] = df['icon'].replace('snow', 6)
    df['icon'] = df['icon'].replace('rain', 5)
    df['icon'] = df['icon'].replace('partly-cloudy-day', 4)
    df['icon'] = df['icon'].replace('partly-cloudy-night', 3)
    df['icon'] = df['icon'].replace('cloudy', 2)
    df['icon'] = df['icon'].replace('wind', 1)
    # cloudCover
    df['cloudCover'] = df['cloudCover'].replace('cloudCover', 0.4)
    df['cloudCover'] = df['cloudCover'].astype('float')
    # summary
    df['summary'] = df['summary'].replace('Rain', 18)
    df['summary'] = df['summary'].replace('Foggy', 17)
    df['summary'] = df['summary'].replace('Flurries and Breezy', 16)
    df['summary'] = df['summary'].replace('Rain and Breezy', 15)
    df['summary'] = df['summary'].replace('Flurries', 14)
    df['summary'] = df['summary'].replace('Clear', 13)
    df['summary'] = df['summary'].replace('Light Snow', 12)
    df['summary'] = df['summary'].replace('Partly Cloudy', 11)
    df['summary'] = df['summary'].replace('Light Rain', 10)
    df['summary'] = df['summary'].replace('Overcast', 9)
    df['summary'] = df['summary'].replace('Mostly Cloudy', 8)
    df['summary'] = df['summary'].replace('Drizzle', 7)
    df['summary'] = df['summary'].replace('Snow', 6)
    df['summary'] = df['summary'].replace('Breezy and Mostly Cloudy', 5)
    df['summary'] = df['summary'].replace('Breezy', 4)
    df['summary'] = df['summary'].replace('Breezy and Partly Cloudy', 3)
    df['summary'] = df['summary'].replace('Heavy Snow', 2)
    df['summary'] = df['summary'].replace('Dry', 1)
    # last row
    df['Furnace'] = df[['Furnace1', 'Furnace2']].sum(axis=1)
    df['Kitchen'] = df[['Kitchen12', 'Kitchen14', 'Kitchen38']].sum(axis=1)
    df.drop(['Furnace1', 'Furnace2', 'Kitchen12', 'Kitchen14', 'Kitchen38'], axis=1, inplace=True)
    # time
    pd.to_datetime(df['time'], unit='s').head(3)
    df['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(df), freq='min'))
    df['month'] = df['time'].apply(lambda x: x.month)
    df['monthday'] = df['time'].apply(lambda x: x.day)
    df['weekday'] = df['time'].apply(lambda x: x.day_name())
    df['hour'] = df['time'].apply(lambda x: x.hour)
    df['minute'] = df['time'].apply(lambda x: x.minute)

    df['weekday'] = df['weekday'].replace('Monday', 1)
    df['weekday'] = df['weekday'].replace('Tuesday', 2)
    df['weekday'] = df['weekday'].replace('Wednesday', 3)
    df['weekday'] = df['weekday'].replace('Thursday', 4)
    df['weekday'] = df['weekday'].replace('Friday', 5)
    df['weekday'] = df['weekday'].replace('Saturday', 6)
    df['weekday'] = df['weekday'].replace('Sunday', 7)

    return df


def df_stats(dataset):
    df_statistics = pd.DataFrame()
    df_statistics['mean'] = dataset.mean()
    df_statistics['median'] = dataset.median()
    df_statistics['max'] = dataset.max()
    df_statistics['min'] = dataset.min()
    df_statistics['count'] = dataset.count()
    df_statistics['std'] = dataset.std()

    a = []
    c1 = stats.linregress(dataset['use'], dataset['use'])
    # slope, intercept, r_value, p_value, std_err = stats.linregress(df['Duration_of_Credit_month'], df['Creditability'])
    c2 = stats.linregress(dataset['Dishwasher'], dataset['use'])
    c3 = stats.linregress(dataset['Home office'], dataset['use'])
    c4 = stats.linregress(dataset['Fridge'], dataset['use'])
    c5 = stats.linregress(dataset['Wine cellar'], dataset['use'])
    c6 = stats.linregress(dataset['Garage door'], dataset['use'])
    c7 = stats.linregress(dataset['Barn'], dataset['use'])
    c8 = stats.linregress(dataset['Well'], dataset['use'])
    c9 = stats.linregress(dataset['Microwave'], dataset['use'])
    c10 = stats.linregress(dataset['Living room'], dataset['use'])
    c11 = stats.linregress(dataset['Solar'], dataset['use'])
    c12 = stats.linregress(dataset['temperature'], dataset['use'])
    c13 = stats.linregress(dataset['icon'], dataset['use'])
    c14 = stats.linregress(dataset['humidity'], dataset['use'])
    c15 = stats.linregress(dataset['visibility'], dataset['use'])
    c16 = stats.linregress(dataset['summary'], dataset['use'])
    c17 = stats.linregress(dataset['pressure'], dataset['use'])
    c18 = stats.linregress(dataset['windSpeed'], dataset['use'])
    c19 = stats.linregress(dataset['cloudCover'], dataset['use'])
    c20 = stats.linregress(dataset['windBearing'], dataset['use'])
    c21 = stats.linregress(dataset['precipProbability'], dataset['use'])
    c22 = stats.linregress(dataset['Furnace'], dataset['use'])
    c23 = stats.linregress(dataset['Kitchen'], dataset['use'])
    #c24 = stats.linregress(dataset['month'], dataset['use'])
    #c25 = stats.linregress(dataset['monthday'], dataset['use'])
    #c26 = stats.linregress(dataset['weekday'], dataset['use'])
    c27 = stats.linregress(dataset['hour'], dataset['use'])
    c28 = stats.linregress(dataset['minute'], dataset['use'])

    p = 3
    a = ["{:.6f}".format(float(c1[p])), c2[p], c3[p], c4[p], c5[p], c6[p], c7[p], c8[p], c9[p], c10[p], c11[p], c12[p],
         c13[p], c14[p], c15[p], c16[p], c17[p], c18[p], c19[p], c20[p], c21[p], c22[p], c23[p], 1, 1, 1, c27[p], c28[p]]
    df_statistics['pval'] = a

    return df_statistics


def df_stats0(X_testt):
    df_statistics = pd.DataFrame()
    df_statistics['mean'] = X_testt.mean()
    df_statistics['median'] = X_testt.median()
    df_statistics['max'] = X_testt.max()
    df_statistics['min'] = X_testt.min()
    df_statistics['count'] = X_testt.count()
    df_statistics['std'] = X_testt.std()

    return df_statistics


def correlation(dataset):
    corr_mlr = dataset.corr(method='pearson')
    plt.figure(figsize=(20, 6))
    sns.heatmap(corr_mlr, annot=True, cmap='Blues')
    plt.title('Correlation matrix')
    plt.show()

    return corr_mlr


def predictions(data, i, X_testt, df_statistics):
    rlog = []; rsvc = []; rknn = []; rgaussian = []; rperception = []; rlinear_svc = []; rsgd = []; rdecision_tree = []; rrandom_forest = []; rxgb = []; rbag = []; rada = []
    rxgb2 = []; rdecision_tree2 = []; rrandom_forest2 = []; rada2 = []; rextra2 = []; rknn2 = []; rcat2 = []; rgradient2 = []; rbag2 = []

    df_train = data.head(i)
    Datum = pd.DataFrame()
    Datum['time'] = df_train['time']
    Y_train = df_train['use'].astype('int')
    del df_train['use'], df_train['time']
    X_train = df_train.astype('int')
    df_test = X_testt
    x = df_test['time']
    del df_test['time']

    '''abhängige Variable'''
    Y_test = df_test['use']
    del df_test['use']
    ''''''

    X_test = df_test
    print('Fortschritt: ' + str(round(0 * 100, 1)) + '%')

    # create an xgboost regression model
    xgb2 = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    xgb2.fit(X_train.values, Y_train.values)
    decision_tree2 = DecisionTreeRegressor()
    decision_tree2.fit(X_train.values, Y_train.values)
    random_forest2 = RandomForestRegressor()
    random_forest2.fit(X_train, Y_train)
    ada2 = AdaBoostRegressor()
    ada2.fit(X_train, Y_train)
    extra2 = ExtraTreesRegressor()
    extra2.fit(X_train, Y_train)
    knn2 = KNeighborsRegressor()
    knn2.fit(X_train, Y_train)
    cat2 = CatBoostRegressor()
    cat2.fit(X_train, Y_train)
    gradient2 = GradientBoostingRegressor()
    gradient2.fit(X_train, Y_train)
    bag2 = BaggingRegressor()
    bag2.fit(X_train, Y_train)



    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(X_train, Y_train)
    df_statistics['mean'] = X_train.mean()
    df_statistics = df_statistics.iloc[1:, :]
    df_statistics['coeff'] = np.around(lin_reg.coef_, 4)
    df_statistics['impact'] = np.around(df_statistics['coeff']*df_statistics['mean'], 4)
    a = []
    '''
    #c1 = stats.linregress(Y_train['use'], Y_train['use'])
    # slope, intercept, r_value, p_value, std_err = stats.linregress(df['Duration_of_Credit_month'], df['Creditability'])
    c2 = stats.linregress(X_train['Dishwasher'], Y_train)
    c3 = 1#stats.linregress(X_train['Home office'], Y_train)
    c4 = 1#stats.linregress(X_train['Fridge'], Y_train)
    c5 = 1#stats.linregress(X_train['Wine cellar'], Y_train)
    c6 = stats.linregress(X_train['Garage door'], Y_train)
    c7 = stats.linregress(X_train['Barn'], Y_train)
    c8 = stats.linregress(X_train['Well'], Y_train)
    c9 = stats.linregress(X_train['Microwave'], Y_train)
    c10 = stats.linregress(X_train['Living room'], Y_train)
    c11 = stats.linregress(X_train['Solar'], Y_train)
    c12 = stats.linregress(X_train['temperature'], Y_train)
    c13 = stats.linregress(X_train['icon'], Y_train)
    c14 = stats.linregress(X_train['humidity'], Y_train)
    c15 = stats.linregress(X_train['visibility'], Y_train)
    c16 = stats.linregress(X_train['summary'], Y_train)
    c17 = stats.linregress(X_train['pressure'], Y_train)
    c18 = stats.linregress(X_train['windSpeed'], Y_train)
    c19 = stats.linregress(X_train['cloudCover'], Y_train)
    c20 = stats.linregress(X_train['windBearing'], Y_train)
    c21 = stats.linregress(X_train['precipProbability'], Y_train)
    c22 = stats.linregress(X_train['Furnace'], Y_train)
    c23 = stats.linregress(X_train['Kitchen'], Y_train)
    # c24 = stats.linregress(X_train['month'], Y_train)
    # c25 = stats.linregress(X_train['monthday'], XY_train)
    # c26 = stats.linregress(X_train['weekday'], Y_train)
    c27 = stats.linregress(X_train['hour'], Y_train)
    c28 = stats.linregress(X_train['minute'], Y_train)
    p = 3
    a = ["{:.6f}".format(float(c2[p])), c3[p], c4[p], c5[p], c6[p], c7[p], c8[p], c9[p], c10[p], c11[p], c12[p],
         c13[p], c14[p], c15[p], c16[p], c17[p], c18[p], c19[p], c20[p], c21[p], c22[p], c23[p], 1, 1, 1, c27[p],
         c28[p]]
    df_statistics['pval'] = a'''
    with open('Infos Trainingset.textmate', 'w') as file:
        file.write(str(df_statistics) + '\n')



    for l in range(0, len(X_test)):
        z = (l+1)/len(X_test)
        print('Fortschritt: ' + str(round(z*100, 1)) + '%')

        prediction_xgb2 = xgb2.predict([X_test.values[l]])
        rxgb2.append(round(prediction_xgb2[0], 3))
        prediction_decision_tree2 = decision_tree2.predict([X_test.values[l]])
        rdecision_tree2.append(round(prediction_decision_tree2[0], 3))
        prediction_random_forest2 = random_forest2.predict([X_test.values[l]])
        rrandom_forest2.append(round(prediction_random_forest2[0], 3))
        prediction_ada2 = ada2.predict([X_test.values[l]])
        rada2.append(round(prediction_ada2[0], 3))
        prediction_extra2 = extra2.predict([X_test.values[l]])
        rextra2.append(round(prediction_extra2[0], 3))
        prediction_knn2 = knn2.predict([X_test.values[l]])
        rknn2.append(round(prediction_knn2[0], 3))
        prediction_cat2 = cat2.predict([X_test.values[l]])
        rcat2.append(round(prediction_cat2[0], 3))
        prediction_gradient2 = gradient2.predict([X_test.values[l]])
        rgradient2.append(round(prediction_gradient2[0], 3))
        prediction_bag2 = bag2.predict([X_test.values[l]])
        rbag2.append(round(prediction_bag2[0], 3))


    df_testresults = pd.DataFrame()
    df_testresults['use'] = Y_test

    df_testresults['xgb2'] = rxgb2
    df_testresults['rdecision_tree2'] = rdecision_tree2
    df_testresults['rrandom_forest2'] = rrandom_forest2
    df_testresults['rada2'] = rada2
    df_testresults['rextra2'] = rextra2
    df_testresults['rextra2'] = rextra2
    df_testresults['rknn2'] = rknn2
    df_testresults['rgradient2'] = rgradient2
    df_testresults['rbag2'] = rbag2

    df_testresults['time'] = x

    return df_testresults


def correlation2(results):
    corr_mlr = results.corr(method='pearson')
    plt.figure(figsize=(20, 6))
    sns.heatmap(corr_mlr, annot=True, cmap='Blues')
    plt.title('Correlation matrix')
    plt.show()

    return corr_mlr


def highrated(results):
    highrated = pd.DataFrame()
    highrated = results
    highrated = highrated.loc[highrated['xgb2'] > -2000]
    #highrated = highrated.loc[highrated['Status'] == 0]
    highrated = highrated.sort_values(by=['evalrextra2'], ascending=False)
    with open('highrated.textmate', 'w') as file:
        file.write(str(highrated) + '\n')

    return highrated


def rmse(results):

    results['evalxgb2'] = abs(results['use'] - results['xgb2'])
    results['evalrdecision_tree2'] = abs(results['use'] - results['rdecision_tree2'])
    results['evalrrandom_forest2'] = abs(results['use'] - results['rrandom_forest2'])
    results['evalrada2'] = abs(results['use'] - results['rada2'])
    results['evalrextra2'] = abs(results['use'] - results['rextra2'])
    results['evalrknn2'] = abs(results['use'] - results['rknn2'])
    results['evalrgradient2'] = abs(results['use'] - results['rgradient2'])
    results['evalrbag2'] = abs(results['use'] - results['rbag2'])

    print('xgb2: ' + str(results['evalxgb2'].mean()))
    print('rdecision_tree2: ' + str(results['evalrextra2'].mean()))
    print('rrandom_forest2: ' + str(results['evalrrandom_forest2'].mean()))
    print('rada2: ' + str(results['evalrada2'].mean()))
    print('rextra2: ' + str(results['evalrextra2'].mean()))
    print('rknn2: ' + str(results['evalrknn2'].mean()))
    print('rgradient2: ' + str(results['evalrgradient2'].mean()))
    print('rbag2: ' + str(results['evalrbag2'].mean()))


main()
