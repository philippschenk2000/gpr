import pandas as pd
import numpy as np
import random as rnd
import json
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
from sklearn import metrics


def main():
    warnings.filterwarnings("ignore")
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    df = pd.read_csv('sunroof_solar.csv', delimiter=',')
    del df['install_size_kw_buckets'], df['center_point'], df['yearly_sunlight_kwh_n'], df['yearly_sunlight_kwh_s']
    del df['yearly_sunlight_kwh_e'], df['yearly_sunlight_kwh_w'], df['yearly_sunlight_kwh_f'], df['yearly_sunlight_kwh_median'], df['kw_total']
    del df['lat_max'], df['lat_min'], df['lng_max'], df['lng_min'], df['state_name']
    del df['number_of_panels_n'], df['number_of_panels_s'], df['number_of_panels_e'], df['number_of_panels_w'], df['number_of_panels_f']
    # corr = correlation(df)
    dataset = df_preprocessing(df)
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    lat_avg = dataset['lat_avg']  # zip_code    lat_avg     lng_avg  count_qualified  percent_covered  percent_qualified  number_of_panels_median  number_of_panels_total  kw_median  yearly_sunlight_kwh_total  carbon_offset_metric_tons  existing_installs_count
    lng_avg = dataset['lng_avg']
    count_qualified = dataset['count_qualified']
    percent_covered = dataset['percent_covered']
    percent_qualified = dataset['percent_qualified']
    number_of_panels_median = dataset['number_of_panels_median']
    number_of_panels_total = dataset['number_of_panels_total']
    kw_median = dataset['kw_median']
    yearly_sunlight_kwh_total = dataset['yearly_sunlight_kwh_total']
    carbon_offset_metric_tons = dataset['carbon_offset_metric_tons']
    existing_installs_count = dataset['existing_installs_count']
    yearly_sunlight_kwh_kw_threshold_avg = df['yearly_sunlight_kwh_kw_threshold_avg']

    with open('Alle Projekte.textmate', 'w') as file:
        file.write(str(dataset) + '\n')
    del dataset['lat_avg'], dataset['lng_avg'], dataset['count_qualified'], dataset['percent_covered'], dataset['percent_qualified']
    del dataset['number_of_panels_median'], dataset['number_of_panels_total'], dataset['kw_median'], dataset['yearly_sunlight_kwh_total']
    del dataset['carbon_offset_metric_tons'], dataset['existing_installs_count'], df['yearly_sunlight_kwh_kw_threshold_avg']


    Trainingsvolumen = 5500
    Testvolumen = 2000


    X_testt = (dataset.head(Trainingsvolumen + Testvolumen)).tail(Testvolumen)
    dataset = dataset.head(Trainingsvolumen)
    with open('Testset.textmate', 'w') as file:
        file.write(str(X_testt) + '\n')
    with open('Trainingset.textmate', 'w') as file:
        file.write(str(dataset) + '\n')
    df_statistics = df_stats(dataset)
    df_statistics0 = df_stats0(X_testt)
    with open('Infos Testset.textmate', 'w') as file:
        file.write(str(df_statistics0) + '\n')
    # corr = correlation(dataset)

    i = int(len(dataset))
    results = pd.DataFrame()
    for x in range(0, 1):
        data = dataset
        # data = dataset.sample(frac=1).reset_index(drop=True)
        # lat_avg
        results = predictions(data, i, X_testt, df_statistics, Trainingsvolumen, Testvolumen, lat_avg, lng_avg, count_qualified, percent_covered, percent_qualified, number_of_panels_median, number_of_panels_total, kw_median, yearly_sunlight_kwh_total, carbon_offset_metric_tons, existing_installs_count)  #xgb
        with open('df_results.textmate', 'w') as file:
            file.write(str(results) + '\n')
        # lng_avg
        results2 = predictions2(data, i, X_testt, df_statistics, Trainingsvolumen, Testvolumen, lat_avg, lng_avg, count_qualified, percent_covered, percent_qualified, number_of_panels_median, number_of_panels_total, kw_median, yearly_sunlight_kwh_total, carbon_offset_metric_tons, existing_installs_count)  #
        with open('df_results2.textmate', 'w') as file:
            file.write(str(results2) + '\n')
        # rest
        results3 = predictions3(data, i, X_testt, df_statistics, results, results2, Trainingsvolumen, Testvolumen, lat_avg, lng_avg,count_qualified, percent_covered, percent_qualified, number_of_panels_median,number_of_panels_total, kw_median, yearly_sunlight_kwh_total, carbon_offset_metric_tons,existing_installs_count, yearly_sunlight_kwh_kw_threshold_avg)  #
        with open('df_results3.textmate', 'w') as file:
            file.write(str(results3) + '\n')
        # count_qualified: 3300; 21, percent_covered: 16; 3,5, percent_qualified: 7,5; 0,1, number_of_panels_median: 23; 0,38, number_of_panels_total: 260000; 26
        # kw_median: 5,5; 0,4, yearly_sunlight_kwh_total: 79055580; 28, carbon_offset_metric_tons: 23; 0,38,

    results = results3
    corr = correlation2(results)
    mse = rmse(results)
    highrate = highrated(mse)


def df_preprocessing(df):
    print('Abschlusswahrscheinlichkeit wird berechnet...')
    # df = df.sort_values(by=['zip_code'], ascending=False)
    df = df.fillna(method="backfill")
    df = df.fillna(df.mean())
    with open('Alle Projekte.textmate', 'w') as file:
        file.write(str(df) + '\n')

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
    # zip_code, state_name, lat_max, lat_min, lng_max, lng_min, lat_avg, lng_avg, yearly_sunlight_kwh_kw_threshold_avg, count_qualified, percent_covered,
    # percent_qualified, number_of_panels_n, number_of_panels_s, number_of_panels_e, number_of_panels_w, number_of_panels_f, number_of_panels_median,
    # number_of_panels_total, kw_median, kw_total, yearly_sunlight_kwh_n, yearly_sunlight_kwh_s, yearly_sunlight_kwh_e, yearly_sunlight_kwh_w, yearly_sunlight_kwh_f,
    # yearly_sunlight_kwh_median, yearly_sunlight_kwh_total, install_size_kw_buckets, carbon_offset_metric_tons, existing_installs_count, center_point

    '''x = dataset['percent_qualified']
    c1 = stats.linregress(dataset['zip_code'], dataset['zip_code'])
    # slope, intercept, r_value, p_value, std_err = stats.linregress(df['Duration_of_Credit_month'], df['Creditability'])
    #c2 = stats.linregress(dataset['state_name'], x)
    c3 = stats.linregress(dataset['lat_max'], x)
    c4 = stats.linregress(dataset['lat_min'], x)
    c5 = stats.linregress(dataset['lng_max'], x)
    c6 = stats.linregress(dataset['lng_min'], x)
    c7 = stats.linregress(dataset['lat_avg'], x)
    c8 = stats.linregress(dataset['lng_avg'], x)
    c9 = stats.linregress(dataset['yearly_sunlight_kwh_kw_threshold_avg'], x)
    c10 = stats.linregress(dataset['count_qualified'], x)
    c11 = stats.linregress(dataset['percent_covered'], x)
    c12 = stats.linregress(dataset['percent_qualified'], x)
    c13 = stats.linregress(dataset['number_of_panels_n'], x)
    c14 = stats.linregress(dataset['number_of_panels_s'], x)
    c15 = stats.linregress(dataset['number_of_panels_e'], x)
    c16 = stats.linregress(dataset['number_of_panels_w'], x)
    c17 = stats.linregress(dataset['number_of_panels_f'], x)
    c18 = stats.linregress(dataset['number_of_panels_median'], x)
    c19 = stats.linregress(dataset['number_of_panels_total'], x)
    c20 = stats.linregress(dataset['kw_median'], x)
    c21 = stats.linregress(dataset['kw_total'], x)
    c22 = stats.linregress(dataset['yearly_sunlight_kwh_n'], x)
    c23 = stats.linregress(dataset['yearly_sunlight_kwh_s'], x)
    c24 = stats.linregress(dataset['yearly_sunlight_kwh_e'], x)
    c25 = stats.linregress(dataset['yearly_sunlight_kwh_w'], x)
    c26 = stats.linregress(dataset['yearly_sunlight_kwh_f'], x)
    c27 = stats.linregress(dataset['yearly_sunlight_kwh_median'], x)
    c28 = stats.linregress(x, x)
    c29 = stats.linregress(dataset['carbon_offset_metric_tons'], x)
    c30 = stats.linregress(dataset['existing_installs_count'], x)

    p = 3
    a = ["{:.6f}".format(float(c1[p])), c3[p], c4[p], c5[p], c6[p], c7[p], c8[p], c9[p], c10[p], c11[p], c12[p],
         c13[p], c14[p], c15[p], c16[p], c17[p], c18[p], c19[p], c20[p], c21[p], c22[p], c23[p], c24[p], c25[p], c26[p], c27[p], c28[p], c29[p], c30[p]]
    df_statistics['pval'] = a'''

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


def correlation(df):
    corr_mlr = df.corr(method='pearson')
    plt.figure(figsize=(20, 6))
    sns.heatmap(corr_mlr, annot=True, cmap='Blues')
    plt.title('Correlation matrix')
    plt.show()

    return corr_mlr


def predictions(data, i, X_testt, df_statistics, Trainingsvolumen, Testvolumen, lat_avg, lng_avg, count_qualified, percent_covered, percent_qualified, number_of_panels_median, number_of_panels_total, kw_median, yearly_sunlight_kwh_total, carbon_offset_metric_tons, existing_installs_count):
    rlog = [];rsvc = [];rknn = [];rgaussian = [];rperception = [];rlinear_svc = [];rsgd = [];rdecision_tree = [];rrandom_forest = [];rxgb = [];rbag = [];rada = []
    rxgb2 = [];rdecision_tree2 = [];rrandom_forest2 = [];rada2 = [];rextra2 = [];rknn2 = [];rcat2 = [];rgradient2 = [];rbag2 = []


    y = lat_avg
    df_train = data.head(i)
    df_train['lat_avg'] = (y.head(Trainingsvolumen))

    '''abhängige Variable'''
    Y_train = df_train['lat_avg'].astype('int')
    del df_train['lat_avg']
    ''''''

    X_train = df_train.astype('int')
    df_test = X_testt
    df_test['lat_avg'] = (y.head(Trainingsvolumen+Testvolumen)).tail(Testvolumen)
    #del df_test['lat_avg']

    '''abhängige Variable'''
    Y_test = df_test['lat_avg']
    del df_test['lat_avg']
    ''''''

    X_test = df_test
    print('Fortschritt: ' + str(round(66.6 + 0 * 33, 1)) + '%')

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

    for l in range(0, len(X_test)):
        z = (l + 1) / len(X_test)
        print('Fortschritt: ' + str(round(66.6 + z * 33, 1)) + '%')

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
    df_testresults['abh. Var'] = Y_test

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


def predictions2(data, i, X_testt, df_statistics, Trainingsvolumen, Testvolumen, lat_avg, lng_avg, count_qualified, percent_covered, percent_qualified, number_of_panels_median, number_of_panels_total, kw_median, yearly_sunlight_kwh_total, carbon_offset_metric_tons, existing_installs_count):
    rlog = [];rsvc = [];rknn = [];rgaussian = [];rperception = [];rlinear_svc = [];rsgd = [];rdecision_tree = [];rrandom_forest = [];rxgb = [];rbag = [];rada = []
    rxgb2 = [];rdecision_tree2 = [];rrandom_forest2 = [];rada2 = [];rextra2 = [];rknn2 = [];rcat2 = [];rgradient2 = [];rbag2 = []


    y = lng_avg
    df_train = data.head(i)
    df_train['lng_avg'] = (y.head(Trainingsvolumen))

    '''abhängige Variable'''
    Y_train = df_train['lng_avg'].astype('int')
    del df_train['lng_avg']
    ''''''

    X_train = df_train.astype('int')
    df_test = X_testt
    df_test['lng_avg'] = (y.head(Trainingsvolumen+Testvolumen)).tail(Testvolumen)
    #del df_test['lat_avg']

    '''abhängige Variable'''
    Y_test = df_test['lng_avg']
    del df_test['lng_avg']
    ''''''

    X_test = df_test
    print('Fortschritt: ' + str(round(66.6 + 0 * 33, 1)) + '%')

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

    for l in range(0, len(X_test)):
        z = (l + 1) / len(X_test)
        print('Fortschritt: ' + str(round(66.6 + z * 33, 1)) + '%')

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
    df_testresults['abh. Var'] = Y_test

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


def predictions3(data, i, X_testt, df_statistics, results, results2, Trainingsvolumen, Testvolumen, lat_avg, lng_avg, count_qualified, percent_covered, percent_qualified, number_of_panels_median, number_of_panels_total, kw_median, yearly_sunlight_kwh_total, carbon_offset_metric_tons, existing_installs_count, yearly_sunlight_kwh_kw_threshold_avg):
    rlog = [];rsvc = [];rknn = [];rgaussian = [];rperception = [];rlinear_svc = [];rsgd = [];rdecision_tree = [];rrandom_forest = [];rxgb = [];rbag = [];rada = []
    rxgb2 = [];rdecision_tree2 = [];rrandom_forest2 = [];rada2 = [];rextra2 = [];rknn2 = [];rcat2 = [];rgradient2 = [];rbag2 = []


    y = carbon_offset_metric_tons
    df_train = data.head(i)
    df_train['lat_avg'] = lat_avg.head(Trainingsvolumen)
    df_train['lng_avg'] = lng_avg.head(Trainingsvolumen)
    df_train['count_qualified'] = (y.head(Trainingsvolumen))


    '''abhängige Variable'''
    Y_train = df_train['count_qualified'].astype('int')
    del df_train['count_qualified']
    ''''''

    X_train = df_train.astype('int')
    df_test = X_testt
    df_test['lat_avg'] = (results['xgb2'].head(Trainingsvolumen+Testvolumen)).tail(Testvolumen)
    df_test['lng_avg'] = (results2['xgb2'].head(Trainingsvolumen+Testvolumen)).tail(Testvolumen)
    df_test['count_qualified'] = (y.head(Trainingsvolumen+Testvolumen)).tail(Testvolumen)

    '''abhängige Variable'''
    Y_test = df_test['count_qualified']
    del df_test['count_qualified']
    ''''''

    X_test = df_test
    print('Fortschritt: ' + str(round(66.6 + 0 * 33, 1)) + '%')

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

    for l in range(0, len(X_test)):
        z = (l + 1) / len(X_test)
        print('Fortschritt: ' + str(round(66.6 + z * 33, 1)) + '%')

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
    df_testresults['abh. Var'] = Y_test

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


def highrated(mse):
    highrated = pd.DataFrame()
    highrated = mse
    highrated = highrated.loc[highrated['xgb2'] > -2000]
    highrated = highrated.sort_values(by=['evalrextra2'], ascending=False)
    with open('highrated.textmate', 'w') as file:
        file.write(str(highrated) + '\n')

    return highrated


def rmse(results):
    results = results
    results['evalxgb2'] = abs(results['abh. Var'] - results['xgb2'])
    results['evalrdecision_tree2'] = abs(results['abh. Var'] - results['rdecision_tree2'])
    results['evalrrandom_forest2'] = abs(results['abh. Var'] - results['rrandom_forest2'])
    results['evalrada2'] = abs(results['abh. Var'] - results['rada2'])
    results['evalrextra2'] = abs(results['abh. Var'] - results['rextra2'])
    results['evalrknn2'] = abs(results['abh. Var'] - results['rknn2'])
    results['evalrgradient2'] = abs(results['abh. Var'] - results['rgradient2'])
    results['evalrbag2'] = abs(results['abh. Var'] - results['rbag2'])

    results['xgb%'] = results['evalxgb2'] / results['abh. Var']
    results['decision_tree%'] = results['evalrdecision_tree2'] / results['abh. Var']
    results['random_forest%'] = results['evalrrandom_forest2'] / results['abh. Var']
    results['ada%'] = results['evalrada2'] / results['abh. Var']
    results['extra%'] = results['evalrextra2'] / results['abh. Var']
    results['knn%'] = results['evalrknn2'] / results['abh. Var']
    results['gradient%'] = results['evalrgradient2'] / results['abh. Var']
    results['bag%'] = results['evalrbag2'] / results['abh. Var']

    results.replace([np.inf, -np.inf], np.nan, inplace=True)
    results.fillna(results.mean)

    print('xgb2: ' + str(results['evalxgb2'].mean()))
    print('rdecision_tree2: ' + str(results['evalrextra2'].mean()))
    print('rrandom_forest2: ' + str(results['evalrrandom_forest2'].mean()))
    print('rada2: ' + str(results['evalrada2'].mean()))
    print('rextra2: ' + str(results['evalrextra2'].mean()))
    print('rknn2: ' + str(results['evalrknn2'].mean()))
    print('rgradient2: ' + str(results['evalrgradient2'].mean()))
    print('rbag2: ' + str(results['evalrbag2'].mean()))

    # print(results['xgb%'].mean())
    print('xgb2: ' + str(results['xgb%'].mean()))
    print('rdecision_tree2: ' + str(results['decision_tree%'].mean()))
    print('rrandom_forest2: ' + str(results['random_forest%'].mean()))
    print('rada2: ' + str(results['ada%'].mean()))
    print('rextra2: ' + str(results['extra%'].mean()))
    print('rknn2: ' + str(results['knn%'].mean()))
    print('rgradient2: ' + str(results['gradient%'].mean()))
    print('rbag2: ' + str(results['bag%'].mean()))

    return results


main()