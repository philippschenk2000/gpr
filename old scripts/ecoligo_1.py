import pandas as pd
import numpy as np
import random as rnd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
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
    #df = pd.read_csv('/Users/psc/PycharmProjects/pythonProject/test_datasets/All_events_per_day_by_platform.csv', delimiter=',')
    df = pd.read_csv('/test_datasets/export1.csv', delimiter=';')
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    with open('../Alle Projekte.textmate', 'w') as file:
        file.write(str(df) + '\n')
    date = df['Date']
    name = df['Name']
    dataset = pd.DataFrame()
    dataset = df_preprocessing(df)
    #dataset = dataset.loc[dataset['Name_GLS Crowd'] == 1]
    dataset = dataset.loc[dataset['month'] > 2]
    dataset = dataset.loc[dataset['month'] < 9]

    #pro Name nochmal, vorsicht bei der freq: H!#
    Trainingsvolumen = 100
    Testvolumen = 50

    del df['Date'], df['Event']
    X_testt = (dataset.head(Trainingsvolumen+Testvolumen)).tail(Testvolumen)
    dataset = dataset.head(Trainingsvolumen)
    with open('Testset.textmate', 'w') as file:
        file.write(str(X_testt) + '\n')
    with open('Trainingset.textmate', 'w') as file:
        file.write(str(dataset) + '\n')

    df_statistics = df_stats(dataset)
    df_statistics0 = df_stats0(X_testt)
    #corr = correlation(dataset)


    i = int(len(dataset))
    results = pd.DataFrame()
    highrate = pd.DataFrame()

    for x in range(0, 1):
        data = dataset
        results = predictions(data, i, X_testt, df_statistics, df_statistics0, date, name)
    corr = correlation2(results)
    with open('df_results.textmate', 'w') as file:
        file.write(str(results) + '\n')
    mse = rmse(results)
    highrate = highrated(results)
    #highrate.to_csv('Höchste Bewertungen 2020.csv')


def df_preprocessing(df):

    print('Abschlusswahrscheinlichkeit wird berechnet...')

    #print(df[["use", "summary"]].groupby(['summary'], as_index=False).mean().sort_values(by='use',ascending=False))
    #df = df[0:-1]
    # time
    #pd.to_datetime(df['Date'], unit='s').head(3)
    #df['Date'] = pd.DatetimeIndex(pd.date_range('2022-06-25T00:00:00', end='2022-09-23T17:00:00', freq='23066350U'))
    df['Date'] = pd.DatetimeIndex(pd.date_range('2021-09-28', end='2022-09-24', freq='403738318U'))#403738318U
    df['month'] = df['Date'].apply(lambda x: x.month)
    df['monthday'] = df['Date'].apply(lambda x: x.day)
    df['weekday'] = df['Date'].apply(lambda x: x.day_name())
    #df['hour'] = df['Date'].apply(lambda x: x.hour)
    #df['minute'] = df['Date'].apply(lambda x: x.minute)

    '''df['weekday'] = df['weekday'].replace('Monday', 1)
    df['weekday'] = df['weekday'].replace('Tuesday', 2)
    df['weekday'] = df['weekday'].replace('Wednesday', 3)
    df['weekday'] = df['weekday'].replace('Thursday', 4)
    df['weekday'] = df['weekday'].replace('Friday', 5)
    df['weekday'] = df['weekday'].replace('Saturday', 6)
    df['weekday'] = df['weekday'].replace('Sunday', 7)'''
    X_cat = pd.get_dummies(df[['Name']], drop_first=True)
    X_catt = pd.get_dummies(df[['weekday']], drop_first=True)
    #df = pd.concat([df['Events'], df['month'], df['monthday'], df['weekday'], df['hour'], X_cat], axis=1)
    df = pd.concat([df['Events'], df['month'], df['monthday'], X_catt, X_cat], axis=1)

    #print(df.head(5))

    return df


def df_stats(dataset):
    df_statistics = pd.DataFrame()
    df_statistics['mean'] = dataset.mean()
    df_statistics['median'] = dataset.median()
    df_statistics['max'] = dataset.max()
    df_statistics['min'] = dataset.min()
    df_statistics['count'] = dataset.count()
    df_statistics['std'] = dataset.std()

    '''
    x = dataset[dataset.columns[0]]
    # slope, intercept, r_value, p_value, std_err = stats.linregress(df['Duration_of_Credit_month'], x)
    a = []

    for i in list(dataset.columns.values):
        c1 = stats.linregress(x, x)
        c2 = stats.linregress(dataset['Name_AK Anlage und Kapital Deutschland AG'], x)
        c3 = stats.linregress(dataset[dataset.columns[7]])
        c4 = stats.linregress(dataset[1], x)
        c5 = stats.linregress(dataset['Wine cellar'], x)
        c6 = stats.linregress(dataset['Garage door'], x)
        c7 = stats.linregress(dataset['Barn'], x)
        c8 = stats.linregress(dataset['Well'], x)
        c9 = stats.linregress(dataset['Microwave'], x)
        c10 = stats.linregress(dataset['Living room'], x)
        c11 = stats.linregress(dataset['Solar'], x)
        c12 = stats.linregress(dataset['temperature'], x)
        c13 = stats.linregress(dataset['icon'], x)
        c14 = stats.linregress(dataset['humidity'], x)
        c15 = stats.linregress(dataset['visibility'], x)
        c16 = stats.linregress(dataset['summary'], x)
        c17 = stats.linregress(dataset['pressure'], x)
        c18 = stats.linregress(dataset['windSpeed'], x)
        c19 = stats.linregress(dataset['cloudCover'], x)
        c20 = stats.linregress(dataset['windBearing'], x)
        c21 = stats.linregress(dataset['precipProbability'], x)
        c22 = stats.linregress(dataset['Furnace'], x)
        c23 = stats.linregress(dataset['Kitchen'], x)
        c24 = stats.linregress(dataset['month'], x)
        c25 = stats.linregress(dataset['monthday'], x)
        c26 = stats.linregress(dataset['weekday'], x)
        c27 = stats.linregress(dataset['hour'], x)
        c28 = stats.linregress(dataset['minute'], x)

    p = 3
    a = ["{:.6f}".format(float(c1[p])), c2[p], c3[p], c4[p], c5[p], c6[p], c7[p], c8[p], c9[p], c10[p], c11[p], c12[p],
         c13[p], c14[p], c15[p], c16[p], c17[p], c18[p], c19[p], c20[p], c21[p], c22[p], c23[p], c24[p], c25[p], c26[p], c27[p], c28[p]]
    df_statistics['pval'] = a
    p = 2
    a = ["{:.6f}".format(float(c1[p])), c2[p], c3[p], c4[p], c5[p], c6[p], c7[p], c8[p], c9[p], c10[p], c11[p], c12[p],
         c13[p], c14[p], c15[p], c16[p], c17[p], c18[p], c19[p], c20[p], c21[p], c22[p], c23[p], c24[p], c25[p], c26[p], c27[p],
         c28[p]]
    df_statistics['r_value'] = a
    '''

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


def predictions(data, i, X_testt, df_statistics, df_statistics0, date, name):
    rlog = []; rsvc = []; rknn = []; rgaussian = []; rperception = []; rlinear_svc = []; rsgd = []; rdecision_tree = []; rrandom_forest = []; rxgb = []; rbag = []; rada = []
    rxgb2 = []; rdecision_tree2 = []; rrandom_forest2 = []; rada2 = []; rextra2 = []; rknn2 = []; rcat2 = []; rgradient2 = []; rbag2 = []

    df_train = data.head(i)
    Datum = pd.DataFrame()
    Y_train = df_train['Events'].astype('int')
    del df_train['Events']
    X_train = df_train.astype('int')
    df_test = X_testt

    '''abhängige Variable'''
    Y_test = df_test['Events']
    del df_test['Events']
    ''''''

    X_test = df_test
    print('Fortschritt: ' + str(round(0 * 100, 1)) + '%')

    # create a xgboost regression model
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
    gradient2 = GradientBoostingRegressor()
    gradient2.fit(X_train, Y_train)
    bag2 = BaggingRegressor()
    bag2.fit(X_train, Y_train)

    logreg = LogisticRegression()
    logreg.fit(X_train.values, Y_train.values)
    svc = SVC()
    svc.fit(X_train.values, Y_train.values)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train.values, Y_train.values)
    gaussian = GaussianNB()
    gaussian.fit(X_train.values, Y_train.values)
    perceptron = Perceptron()
    perceptron.fit(X_train.values, Y_train.values)
    linear_svc = LinearSVC()
    linear_svc.fit(X_train.values, Y_train.values)
    sgd = SGDClassifier()
    sgd.fit(X_train.values, Y_train.values)
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train.values, Y_train.values)
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train.values, Y_train.values)
    #xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)
    #xgb.fit(X_train.values, Y_train.values)
    #bag = BaggingClassifier()
    #bag.fit(X_train.values, Y_train.values)
    ada = AdaBoostClassifier()
    ada.fit(X_train.values, Y_train.values)


    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(X_train, Y_train)
    df_statistics['mean'] = X_train.mean()
    df_statistics = df_statistics.iloc[1:, :]
    df_statistics['coeff'] = np.around(lin_reg.coef_, 4)
    df_statistics['impact'] = np.around(df_statistics['coeff']*df_statistics['mean'], 4)
    with open('Infos Trainingset.textmate', 'w') as file:
        file.write(str(df_statistics) + '\n')

    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(X_test, Y_test)
    df_statistics0['mean'] = X_test.mean()
    df_statistics0 = df_statistics0.iloc[1:, :]
    df_statistics0['coeff'] = np.around(lin_reg.coef_, 4)
    df_statistics0['impact'] = np.around(df_statistics0['coeff'] * df_statistics0['mean'], 4)
    with open('Infos Testset.textmate', 'w') as file:
        file.write(str(df_statistics0) + '\n')

    a = []
    for l in range(0, len(X_test)):

        values = X_test.values[l]
        prediction_log = logreg.predict_proba([values])
        if prediction_log[0][1] > 0.5:
            rlog.append(1)
        else:
            rlog.append(0)
        prediction_svc = svc.predict([values])
        rsvc.append((prediction_svc[0]))
        prediction_knn = knn.predict([values])
        rknn.append((prediction_knn[0]))
        prediction_gaussian = gaussian.predict([values])
        rgaussian.append((prediction_gaussian[0]))
        prediction_perceptron = perceptron.predict([values])
        rperception.append((prediction_perceptron[0]))
        prediction_linear_svc = linear_svc.predict([values])
        rlinear_svc.append((prediction_linear_svc[0]))
        prediction_sgd = sgd.predict([values])
        rsgd.append((prediction_sgd[0]))
        prediction_decision_tree = decision_tree.predict([values])
        rdecision_tree.append((prediction_decision_tree[0]))
        prediction_random_forest = random_forest.predict([values])
        rrandom_forest.append((prediction_random_forest[0]))
        #prediction_xgb = xgb.predict([values])
        #rxgb.append((prediction_xgb[0]))
        #prediction_bag = bag.predict([values])
        #rbag.append((prediction_bag[0]))
        prediction_ada = ada.predict([values])
        rada.append((prediction_ada[0]))

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
        prediction_gradient2 = gradient2.predict([X_test.values[l]])
        rgradient2.append(round(prediction_gradient2[0], 3))
        prediction_bag2 = bag2.predict([X_test.values[l]])
        rbag2.append(round(prediction_bag2[0], 3))


    df_testresults = pd.DataFrame()
    df_testresults['Events'] = Y_test
    df_testresults['Name'] = name
    df_testresults['time'] = date

    df_testresults['log'] = rlog
    df_testresults['svc'] = rsvc
    df_testresults['knn'] = rknn
    df_testresults['gaussian'] = rgaussian
    df_testresults['perception'] = rperception
    df_testresults['linear_svc'] = rlinear_svc
    df_testresults['sgd'] = rsgd
    df_testresults['decision_tree'] = rdecision_tree
    df_testresults['random_forest'] = rrandom_forest
    #df_testresults['xgb'] = rxgb
    #df_testresults['bag'] = rbag
    df_testresults['ada'] = rada

    df_testresults['xgb2'] = rxgb2
    df_testresults['rdecision_tree2'] = rdecision_tree2
    df_testresults['rrandom_forest2'] = rrandom_forest2
    df_testresults['rada2'] = rada2
    df_testresults['rextra2'] = rextra2
    df_testresults['rextra2'] = rextra2
    df_testresults['rknn2'] = rknn2
    df_testresults['rgradient2'] = rgradient2
    df_testresults['rbag2'] = rbag2

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

    x = results['Events']
    results['evalxgb2'] = abs(x - results['xgb2'])
    results['evalrdecision_tree2'] = abs(x - results['rdecision_tree2'])
    results['evalrrandom_forest2'] = abs(x - results['rrandom_forest2'])
    results['evalrada2'] = abs(x - results['rada2'])
    results['evalrextra2'] = abs(x - results['rextra2'])
    results['evalrknn2'] = abs(x - results['rknn2'])
    results['evalrgradient2'] = abs(x - results['rgradient2'])
    results['evalrbag2'] = abs(x - results['rbag2'])

    print('xgb2: ' + str(results['evalxgb2'].mean()))
    print('rdecision_tree2: ' + str(results['evalrextra2'].mean()))
    print('rrandom_forest2: ' + str(results['evalrrandom_forest2'].mean()))
    print('rada2: ' + str(results['evalrada2'].mean()))
    print('rextra2: ' + str(results['evalrextra2'].mean()))
    print('rknn2: ' + str(results['evalrknn2'].mean()))
    print('rgradient2: ' + str(results['evalrgradient2'].mean()))
    print('rbag2: ' + str(results['evalrbag2'].mean()))

    results['xgb%'] = results['evalxgb2'] / x
    results['decision_tree%'] = results['evalrdecision_tree2'] / x
    results['random_forest%'] = results['evalrrandom_forest2'] / x
    results['ada%'] = results['evalrada2'] / x
    results['extra%'] = results['evalrextra2'] / x
    results['knn%'] = results['evalrknn2'] / x
    results['gradient%'] = results['evalrgradient2'] / x
    results['bag%'] = results['evalrbag2'] / x

    results=results.loc[results['Events']>0]
    print(results.head())

    print('xgb2: ' + str(results['xgb%'].mean()))
    print('rdecision_tree2: ' + str(results['decision_tree%'].mean()))
    print('rrandom_forest2: ' + str(results['random_forest%'].mean()))
    print('rada2: ' + str(results['ada%'].mean()))
    print('rextra2: ' + str(results['extra%'].mean()))
    print('rknn2: ' + str(results['knn%'].mean()))
    print('rgradient2: ' + str(results['gradient%'].mean()))
    print('rbag2: ' + str(results['bag%'].mean()))


main()
