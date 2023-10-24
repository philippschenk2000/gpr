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
    del df['install_size_kw_buckets'], df['center_point'], df['yearly_sunlight_kwh_n'], df['yearly_sunlight_kwh_s'], df['yearly_sunlight_kwh_e'], df['yearly_sunlight_kwh_w'], df['yearly_sunlight_kwh_f'], df['yearly_sunlight_kwh_median'], df['kw_total'], df['number_of_panels_total']
    del df['lat_max'], df['lat_min'], df['lng_max'], df['lng_min'], df['state_name']
    #corr = correlation(df)
    dataset = df_preprocessing(df)
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    yearly_sunlight = dataset['yearly_sunlight_kwh_total']
    del dataset['yearly_sunlight_kwh_total']
    with open('Alle Projekte.textmate', 'w') as file:
        file.write(str(dataset) + '\n')


    Trainingsvolumen = 5500
    Testvolumen = 2000


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
    results2 = pd.DataFrame()
    results3 = pd.DataFrame()
    results4 = pd.DataFrame()
    for x in range(0, 1):
        data = dataset
        #data = dataset.sample(frac=1).reset_index(drop=True)

        # lat_avg
        results = predictions(data, i, X_testt, df_statistics) #rextra2
        with open('df_results.textmate', 'w') as file:
            file.write(str(results) + '\n')

        # lng_avg
        results2 = predictions2(data, i, X_testt, df_statistics, results) #rada2
        with open('df_results2.textmate', 'w') as file:
            file.write(str(results2) + '\n')

        # yearly_sunlight_kwh_total
        results3 = predictions3(data, i, X_testt, df_statistics, results, results2, yearly_sunlight, Trainingsvolumen, Testvolumen) #rknn2
        with open('df_results3.textmate', 'w') as file:
            file.write(str(results3) + '\n')

        # carbon_offset_metric_tons
        results4 = predictions4(data, i, X_testt, df_statistics, results, results2, results3, yearly_sunlight, Trainingsvolumen, Testvolumen) #rextra2
        with open('df_results4.textmate', 'w') as file:
            file.write(str(results4) + '\n')

        # count_qualified
        results5 = predictions5(data, i, X_testt, df_statistics, results, results2, results3, results4, yearly_sunlight, Trainingsvolumen, Testvolumen)  # r
        with open('df_results5.textmate', 'w') as file:
            file.write(str(results5) + '\n')

        # percent_covered
        results6 = predictions6(data, i, X_testt, df_statistics, results, results2, results3, results4, results5, yearly_sunlight, Trainingsvolumen, Testvolumen)  # r
        with open('df_results5.textmate', 'w') as file:
            file.write(str(results5) + '\n')

    corr = correlation2(results3)
    mse = rmse(results4)
    highrate = highrated(mse)


def df_preprocessing(df):

    print('Abschlusswahrscheinlichkeit wird berechnet...')
    #df = df.sort_values(by=['zip_code'], ascending=False)
    df = df.fillna(method="backfill")
    df = df.fillna(df.mean())
    with open('Alle Projekte.textmate', 'w') as file:
        file.write(str(df) + '\n')

    '''df = df.append({'Status': 0, 'Herkunft': float(input("Herkunft (?=nan; KOP=1; Eigenakquise=2):")),
                    'Kanal': (input("Kanal (Unbekannt; Internet; Flyer; Andere...):")),
                    'Geschlecht': float(input("Geschlecht (?=nan; Frau=1; Herr=2):")),
                    'Alter': float(input("Alter (?=nan; in Jahren):")),
                    'Kinder': float(input("Kinder (?=nan; Nein=0; Ja=1):")), 
                    'Familienstand': float(input("Familienstand (?=nan; Verheiratet=1; Geschieden/Getrennt=2; Verwitwet=3; Ledig=4):")),
                    'Alter2': float(input("Partner (?=nan; Nein=0; Ja=1):")),
                    'Bundesland': (input("Bundesland (Unbekannt; Hessen; Berlin; Bayern...):")),
                    'Objektart': (input("Objektart (?=nan; Einfamilienhaus; Doppelhaushälfte...):")),
                    'Keller': float(input("Keller (?=nan; Nicht unterkellert=1; Teilunterkellert=2; Unterkellert=3):")),
                    'Objektzustand': float(input("Objektzustand (?=nan; Sanierobjekt=1; Befriedigend=2; Mittel=3; Gut=4):")),
                    'Wohnlage': float(input("Wohnlage (?=nan; Schlecht=1; Mittel=2; Gut=3):")),
                    'Wohnhaft': float(input("Wohnhaft (?=nan; Jahreszahl):")),
                    'Heizung': float(input("Heizung (?=nan; Jahreszahl):")),
                    'Wohnflaeche': float(input("Wohnflaeche (?=nan; In Meter^2):")),
                    'Garagen': float(input("Garagen (?=nan; Nein=0; Ja=1):")),
                    'Stellplätze': float(input("Stellplätze (?=nan; Nein=0; Ja=1):")),
                    'Immobilienwert_intern': float(input("Immobilienwert_intern (?=nan; Höhe):")),
                    'Gutacherwert': float(input("Gutacherwert (?=nan; Höhe):"))}, ignore_index=True)
                    '''

    #print(df[["use", "summary"]].groupby(['summary'], as_index=False).mean().sort_values(by='use',ascending=False))
    '''
    bucket_texas = df['install_size_kw_buckets'].apply(json.loads)
    # converting into list
    buckets_texas = list(bucket_texas)
    # converting into dataframe
    buckpd_texas = pd.DataFrame(buckets_texas).reset_index(drop=True)
    buckpd_texas.dropna(inplace=True)
    buckpd_texas_melt = pd.melt(buckpd_texas)  # converting into one column
    print(buckpd_texas_melt)

    buckpd_texas_melt[['bucket_kW', 'no_of_buildings']] = pd.DataFrame(buckpd_texas_melt.value.values.tolist(),
                                                                       index=buckpd_texas_melt.index)
    print(buckpd_texas_melt)
    # converting into ranges
    bins = pd.cut(buckpd_texas_melt['bucket_kW'], [0, 100, 1000])
    tx = buckpd_texas_melt.groupby(bins)['bucket_kW'].agg(['count', 'sum'])
    tx.columns = ['count_tx', 'sum_tx']
    tx['range'] = ['0 - 100', '100 - 1000']
    print(tx)'''


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
    #zip_code, state_name, lat_max, lat_min, lng_max, lng_min, lat_avg, lng_avg, yearly_sunlight_kwh_kw_threshold_avg, count_qualified, percent_covered,
    #percent_qualified, number_of_panels_n, number_of_panels_s, number_of_panels_e, number_of_panels_w, number_of_panels_f, number_of_panels_median,
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


def predictions(data, i, X_testt, df_statistics):
    rlog = []; rsvc = []; rknn = []; rgaussian = []; rperception = []; rlinear_svc = []; rsgd = []; rdecision_tree = []; rrandom_forest = []; rxgb = []; rbag = []; rada = []
    rxgb2 = []; rdecision_tree2 = []; rrandom_forest2 = []; rada2 = []; rextra2 = []; rknn2 = []; rcat2 = []; rgradient2 = []; rbag2 = []

    df_train = data.head(i)
    Datum = pd.DataFrame()
    #Datum['state_name'] = df_train['state_name']

    '''abhängige Variable'''
    Y_train = df_train['lat_avg'].astype('int')
    del df_train['lat_avg']
    ''''''

    X_train = df_train.astype('int')
    df_test = X_testt

    '''abhängige Variable'''
    Y_test = df_test['lat_avg']
    del df_test['lat_avg']
    ''''''

    X_test = df_test
    print('Fortschritt: ' + str(round(0 * 33, 1)) + '%')

    # create an xgboost regression model
    '''xgb2 = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    xgb2.fit(X_train.values, Y_train.values)
    decision_tree2 = DecisionTreeRegressor()
    decision_tree2.fit(X_train.values, Y_train.values)
    random_forest2 = RandomForestRegressor()
    random_forest2.fit(X_train, Y_train)
    ada2 = AdaBoostRegressor()
    ada2.fit(X_train, Y_train)'''
    extra2 = ExtraTreesRegressor()
    extra2.fit(X_train, Y_train)
    '''knn2 = KNeighborsRegressor()
    knn2.fit(X_train, Y_train)
    cat2 = CatBoostRegressor()
    cat2.fit(X_train, Y_train)
    gradient2 = GradientBoostingRegressor()
    gradient2.fit(X_train, Y_train)
    bag2 = BaggingRegressor()
    bag2.fit(X_train, Y_train)'''



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
        print('Fortschritt: ' + str(round(z*33, 1)) + '%')

        '''prediction_xgb2 = xgb2.predict([X_test.values[l]])
        rxgb2.append(round(prediction_xgb2[0], 3))
        prediction_decision_tree2 = decision_tree2.predict([X_test.values[l]])
        rdecision_tree2.append(round(prediction_decision_tree2[0], 3))
        prediction_random_forest2 = random_forest2.predict([X_test.values[l]])
        rrandom_forest2.append(round(prediction_random_forest2[0], 3))
        prediction_ada2 = ada2.predict([X_test.values[l]])
        rada2.append(round(prediction_ada2[0], 3))'''
        prediction_extra2 = extra2.predict([X_test.values[l]])
        rextra2.append(round(prediction_extra2[0], 3))
        '''prediction_knn2 = knn2.predict([X_test.values[l]])
        rknn2.append(round(prediction_knn2[0], 3))
        prediction_cat2 = cat2.predict([X_test.values[l]])
        rcat2.append(round(prediction_cat2[0], 3))
        prediction_gradient2 = gradient2.predict([X_test.values[l]])
        rgradient2.append(round(prediction_gradient2[0], 3))
        prediction_bag2 = bag2.predict([X_test.values[l]])
        rbag2.append(round(prediction_bag2[0], 3))'''


    df_testresults = pd.DataFrame()
    df_testresults['abh. Var'] = Y_test

    '''df_testresults['xgb2'] = rxgb2
    df_testresults['rdecision_tree2'] = rdecision_tree2
    df_testresults['rrandom_forest2'] = rrandom_forest2
    df_testresults['rada2'] = rada2'''
    df_testresults['rextra2'] = rextra2
    '''df_testresults['rknn2'] = rknn2
    df_testresults['rgradient2'] = rgradient2
    df_testresults['rbag2'] = rbag2'''

    return df_testresults


def predictions2(data, i, X_testt, df_statistics, results):
    rlog = []; rsvc = []; rknn = []; rgaussian = []; rperception = []; rlinear_svc = []; rsgd = []; rdecision_tree = []; rrandom_forest = []; rxgb = []; rbag = []; rada = []
    rxgb2 = []; rdecision_tree2 = []; rrandom_forest2 = []; rada2 = []; rextra2 = []; rknn2 = []; rcat2 = []; rgradient2 = []; rbag2 = []
    df_train = data.head(i)

    '''abhängige Variable'''
    Y_train = df_train['lng_avg'].astype('int')
    del df_train['lng_avg']
    ''''''

    X_train = df_train.astype('int')
    df_test = X_testt
    df_test['lat_avg'] = results['rextra2']

    '''abhängige Variable'''
    Y_test = df_test['lng_avg']
    del df_test['lng_avg']
    ''''''

    X_test = df_test
    print('Fortschritt: ' + str(round(33.3 + 0 * 33, 1)) + '%')

    # create an xgboost regression model
    '''xgb2 = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    xgb2.fit(X_train.values, Y_train.values)
    decision_tree2 = DecisionTreeRegressor()
    decision_tree2.fit(X_train.values, Y_train.values)
    random_forest2 = RandomForestRegressor()
    random_forest2.fit(X_train, Y_train)'''
    ada2 = AdaBoostRegressor()
    ada2.fit(X_train, Y_train)
    '''extra2 = ExtraTreesRegressor()
    extra2.fit(X_train, Y_train)
    knn2 = KNeighborsRegressor()
    knn2.fit(X_train, Y_train)
    cat2 = CatBoostRegressor()
    cat2.fit(X_train, Y_train)
    gradient2 = GradientBoostingRegressor()
    gradient2.fit(X_train, Y_train)
    bag2 = BaggingRegressor()
    bag2.fit(X_train, Y_train)'''



    for l in range(0, len(X_test)):
        z = (l+1)/len(X_test)
        print('Fortschritt: ' + str(round(33.3+z*33, 1)) + '%')

        '''prediction_xgb2 = xgb2.predict([X_test.values[l]])
        rxgb2.append(round(prediction_xgb2[0], 3))
        prediction_decision_tree2 = decision_tree2.predict([X_test.values[l]])
        rdecision_tree2.append(round(prediction_decision_tree2[0], 3))
        prediction_random_forest2 = random_forest2.predict([X_test.values[l]])
        rrandom_forest2.append(round(prediction_random_forest2[0], 3))'''
        prediction_ada2 = ada2.predict([X_test.values[l]])
        rada2.append(round(prediction_ada2[0], 3))
        '''prediction_extra2 = extra2.predict([X_test.values[l]])
        rextra2.append(round(prediction_extra2[0], 3))
        prediction_knn2 = knn2.predict([X_test.values[l]])
        rknn2.append(round(prediction_knn2[0], 3))
        prediction_cat2 = cat2.predict([X_test.values[l]])
        rcat2.append(round(prediction_cat2[0], 3))
        prediction_gradient2 = gradient2.predict([X_test.values[l]])
        rgradient2.append(round(prediction_gradient2[0], 3))
        prediction_bag2 = bag2.predict([X_test.values[l]])
        rbag2.append(round(prediction_bag2[0], 3))'''


    df_testresults = pd.DataFrame()
    df_testresults['abh. Var'] = Y_test

    '''df_testresults['xgb2'] = rxgb2
    df_testresults['rdecision_tree2'] = rdecision_tree2
    df_testresults['rrandom_forest2'] = rrandom_forest2'''
    df_testresults['rada2'] = rada2
    '''df_testresults['rextra2'] = rextra2
    df_testresults['rextra2'] = rextra2
    df_testresults['rknn2'] = rknn2
    df_testresults['rgradient2'] = rgradient2
    df_testresults['rbag2'] = rbag2'''

    return df_testresults


def predictions3(data, i, X_testt, df_statistics, results, results2, yearly_sunlight, Trainingsvolumen, Testvolumen):
    rlog = []; rsvc = []; rknn = []; rgaussian = []; rperception = []; rlinear_svc = []; rsgd = []; rdecision_tree = []; rrandom_forest = []; rxgb = []; rbag = []; rada = []
    rxgb2 = []; rdecision_tree2 = []; rrandom_forest2 = []; rada2 = []; rextra2 = []; rknn2 = []; rcat2 = []; rgradient2 = []; rbag2 = []

    df_train = data.head(i)
    #print(yearly_sunlight.head(Trainingsvolumen))
    df_train['yearly_sunlight_kwh_total'] = yearly_sunlight.head(Trainingsvolumen)
    #print(df_train.tail())

    '''abhängige Variable'''
    Y_train = df_train['yearly_sunlight_kwh_total'].astype('int')
    del df_train['yearly_sunlight_kwh_total']
    ''''''

    X_train = df_train.astype('int')
    df_test = X_testt
    df_test['lat_avg'] = results['rextra2']
    df_test['lng_avg'] = results2['rada2']
    df_test['yearly_sunlight_kwh_total'] = (yearly_sunlight.head(Trainingsvolumen+Testvolumen)).tail(Testvolumen)
    #print(df_test.head())

    '''abhängige Variable'''
    Y_test = df_test['yearly_sunlight_kwh_total']
    del df_test['yearly_sunlight_kwh_total']
    ''''''

    X_test = df_test
    #print(X_test.head())
    print('Fortschritt: ' + str(round(66.6 + 0 * 33, 1)) + '%')

    # create an xgboost regression model
    '''xgb2 = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    xgb2.fit(X_train.values, Y_train.values)
    decision_tree2 = DecisionTreeRegressor()
    decision_tree2.fit(X_train.values, Y_train.values)
    random_forest2 = RandomForestRegressor()
    random_forest2.fit(X_train, Y_train)
    ada2 = AdaBoostRegressor()
    ada2.fit(X_train, Y_train)
    extra2 = ExtraTreesRegressor()
    extra2.fit(X_train, Y_train)'''
    knn2 = KNeighborsRegressor()
    knn2.fit(X_train, Y_train)
    '''cat2 = CatBoostRegressor()
    cat2.fit(X_train, Y_train)
    gradient2 = GradientBoostingRegressor()
    gradient2.fit(X_train, Y_train)
    bag2 = BaggingRegressor()
    bag2.fit(X_train, Y_train)'''



    for l in range(0, len(X_test)):
        z = (l+1)/len(X_test)
        print('Fortschritt: ' + str(round(66.6+z*33, 1)) + '%')

        '''prediction_xgb2 = xgb2.predict([X_test.values[l]])
        rxgb2.append(round(prediction_xgb2[0], 3))
        prediction_decision_tree2 = decision_tree2.predict([X_test.values[l]])
        rdecision_tree2.append(round(prediction_decision_tree2[0], 3))
        prediction_random_forest2 = random_forest2.predict([X_test.values[l]])
        rrandom_forest2.append(round(prediction_random_forest2[0], 3))
        prediction_ada2 = ada2.predict([X_test.values[l]])
        rada2.append(round(prediction_ada2[0], 3))
        prediction_extra2 = extra2.predict([X_test.values[l]])
        rextra2.append(round(prediction_extra2[0], 3))'''
        prediction_knn2 = knn2.predict([X_test.values[l]])
        rknn2.append(round(prediction_knn2[0], 3))
        '''prediction_cat2 = cat2.predict([X_test.values[l]])
        rcat2.append(round(prediction_cat2[0], 3))
        prediction_gradient2 = gradient2.predict([X_test.values[l]])
        rgradient2.append(round(prediction_gradient2[0], 3))
        prediction_bag2 = bag2.predict([X_test.values[l]])
        rbag2.append(round(prediction_bag2[0], 3))'''


    df_testresults = pd.DataFrame()
    df_testresults['abh. Var'] = Y_test

    '''df_testresults['xgb2'] = rxgb2
    df_testresults['rdecision_tree2'] = rdecision_tree2
    df_testresults['rrandom_forest2'] = rrandom_forest2
    #df_testresults['rada2'] = rada2
    df_testresults['rextra2'] = rextra2'''
    df_testresults['rknn2'] = rknn2
    '''df_testresults['rgradient2'] = rgradient2
    df_testresults['rbag2'] = rbag2'''

    return df_testresults


def predictions4(data, i, X_testt, df_statistics, results, results2, results3, yearly_sunlight, Trainingsvolumen, Testvolumen):
    rlog = []; rsvc = []; rknn = []; rgaussian = []; rperception = []; rlinear_svc = []; rsgd = []; rdecision_tree = []; rrandom_forest = []; rxgb = []; rbag = []; rada = []
    rxgb2 = []; rdecision_tree2 = []; rrandom_forest2 = []; rada2 = []; rextra2 = []; rknn2 = []; rcat2 = []; rgradient2 = []; rbag2 = []

    df_train = data.head(i)
    del df_train['yearly_sunlight_kwh_kw_threshold_avg'], df_train['number_of_panels_n'], df_train['number_of_panels_s'], df_train['number_of_panels_e'], df_train['number_of_panels_w'], df_train['number_of_panels_f'], df_train['number_of_panels_median'], df_train['kw_median']#, df_train['carbon_offset_metric_tons']
    df_train['yearly_sunlight_kwh_total'] = yearly_sunlight.head(Trainingsvolumen)
    #print(df_train.tail())

    '''abhängige Variable'''
    Y_train = df_train['carbon_offset_metric_tons'].astype('int')
    del df_train['carbon_offset_metric_tons']
    ''''''

    X_train = df_train.astype('int')
    df_test = X_testt
    df_test['lat_avg'] = results['rextra2']
    df_test['lng_avg'] = results2['rada2']
    df_test['yearly_sunlight_kwh_total'] = results3['rknn2']
    #df_test['yearly_sunlight_kwh_total'] = (yearly_sunlight.head(Trainingsvolumen+Testvolumen)).tail(Testvolumen)
    del df_test['yearly_sunlight_kwh_kw_threshold_avg'], df_test['number_of_panels_n'], df_test['number_of_panels_s'], df_test['number_of_panels_e'], df_test['number_of_panels_w'], df_test['number_of_panels_f'], df_test['number_of_panels_median'], df_test['kw_median']#, df_test['carbon_offset_metric_tons']

    '''abhängige Variable'''
    Y_test = df_test['carbon_offset_metric_tons']
    del df_test['carbon_offset_metric_tons']
    ''''''

    X_test = df_test
    #print(X_test.head())
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
        z = (l+1)/len(X_test)
        print('Fortschritt: ' + str(round(66.6+z*33, 1)) + '%')

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

    #df_testresults['time'] = x

    return df_testresults


def predictions5(data, i, X_testt, df_statistics, results, results2, results3, results4, yearly_sunlight, Trainingsvolumen, Testvolumen):
    rlog = []; rsvc = []; rknn = []; rgaussian = []; rperception = []; rlinear_svc = []; rsgd = []; rdecision_tree = []; rrandom_forest = []; rxgb = []; rbag = []; rada = []
    rxgb2 = []; rdecision_tree2 = []; rrandom_forest2 = []; rada2 = []; rextra2 = []; rknn2 = []; rcat2 = []; rgradient2 = []; rbag2 = []

    df_train = data.head(i)
    del df_train['yearly_sunlight_kwh_kw_threshold_avg'], df_train['number_of_panels_n'], df_train['number_of_panels_s'], df_train['number_of_panels_e'], df_train['number_of_panels_w'], df_train['number_of_panels_f'], df_train['number_of_panels_median'], df_train['kw_median']#, df_train['carbon_offset_metric_tons']
    df_train['yearly_sunlight_kwh_total'] = yearly_sunlight.head(Trainingsvolumen)
    #print(df_train.tail())

    '''abhängige Variable'''
    Y_train = df_train['count_qualified'].astype('int')
    del df_train['count_qualified']
    ''''''

    X_train = df_train.astype('int')
    df_test = X_testt
    df_test['lat_avg'] = results['rextra2']
    df_test['lng_avg'] = results2['rada2']
    df_test['yearly_sunlight_kwh_total'] = results3['rknn2']
    df_test['carbon_offset_metric_tons'] = results4['rextra2']
    #df_test['yearly_sunlight_kwh_total'] = (yearly_sunlight.head(Trainingsvolumen+Testvolumen)).tail(Testvolumen)
    del df_test['yearly_sunlight_kwh_kw_threshold_avg'], df_test['number_of_panels_n'], df_test['number_of_panels_s'], df_test['number_of_panels_e'], df_test['number_of_panels_w'], df_test['number_of_panels_f'], df_test['number_of_panels_median'], df_test['kw_median']#, df_test['carbon_offset_metric_tons']

    '''abhängige Variable'''
    Y_test = df_test['count_qualified']
    del df_test['count_qualified']
    ''''''

    X_test = df_test
    #print(X_test.head())
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
        z = (l+1)/len(X_test)
        print('Fortschritt: ' + str(round(66.6+z*33, 1)) + '%')

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

    #df_testresults['time'] = x

    return df_testresults


def predictions6(data, i, X_testt, df_statistics, results, results2, results3, results4, results5, yearly_sunlight, Trainingsvolumen, Testvolumen):
    rlog = []; rsvc = []; rknn = []; rgaussian = []; rperception = []; rlinear_svc = []; rsgd = []; rdecision_tree = []; rrandom_forest = []; rxgb = []; rbag = []; rada = []
    rxgb2 = []; rdecision_tree2 = []; rrandom_forest2 = []; rada2 = []; rextra2 = []; rknn2 = []; rcat2 = []; rgradient2 = []; rbag2 = []

    df_train = data.head(i)
    del df_train['yearly_sunlight_kwh_kw_threshold_avg'], df_train['number_of_panels_n'], df_train['number_of_panels_s'], df_train['number_of_panels_e'], df_train['number_of_panels_w'], df_train['number_of_panels_f'], df_train['number_of_panels_median'], df_train['kw_median']#, df_train['carbon_offset_metric_tons']
    df_train['yearly_sunlight_kwh_total'] = yearly_sunlight.head(Trainingsvolumen)
    #print(df_train.tail())

    '''abhängige Variable'''
    Y_train = df_train['carbon_offset_metric_tons'].astype('int')
    del df_train['carbon_offset_metric_tons']
    ''''''

    X_train = df_train.astype('int')
    df_test = X_testt
    df_test['lat_avg'] = results['rextra2']
    df_test['lng_avg'] = results2['rada2']
    df_test['yearly_sunlight_kwh_total'] = results3['rknn2']
    df_test['yearly_sunlight_kwh_total'] = (yearly_sunlight.head(Trainingsvolumen+Testvolumen)).tail(Testvolumen)
    del df_test['yearly_sunlight_kwh_kw_threshold_avg'], df_test['number_of_panels_n'], df_test['number_of_panels_s'], df_test['number_of_panels_e'], df_test['number_of_panels_w'], df_test['number_of_panels_f'], df_test['number_of_panels_median'], df_test['kw_median']#, df_test['carbon_offset_metric_tons']

    '''abhängige Variable'''
    Y_test = df_test['carbon_offset_metric_tons']
    del df_test['carbon_offset_metric_tons']
    ''''''

    X_test = df_test
    #print(X_test.head())
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
        z = (l+1)/len(X_test)
        print('Fortschritt: ' + str(round(66.6+z*33, 1)) + '%')

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

    #df_testresults['time'] = x

    return df_testresults


def correlation2(results3):
    corr_mlr = results3.corr(method='pearson')
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


def rmse(results4):
    results = results4
    results['evalxgb2'] = abs(results['abh. Var'] - results['xgb2'])
    results['evalrdecision_tree2'] = abs(results['abh. Var'] - results['rdecision_tree2'])
    results['evalrrandom_forest2'] = abs(results['abh. Var'] - results['rrandom_forest2'])
    results['evalrada2'] = abs(results['abh. Var'] - results['rada2'])
    results['evalrextra2'] = abs(results['abh. Var'] - results['rextra2'])
    results['evalrknn2'] = abs(results['abh. Var'] - results['rknn2'])
    results['evalrgradient2'] = abs(results['abh. Var'] - results['rgradient2'])
    results['evalrbag2'] = abs(results['abh. Var'] - results['rbag2'])

    results['xgb%'] = results['evalxgb2']/results['abh. Var']
    results['decision_tree%'] = results['evalrdecision_tree2']/results['abh. Var']
    results['random_forest%'] = results['evalrrandom_forest2']/results['abh. Var']
    results['ada%'] = results['evalrada2']/results['abh. Var']
    results['extra%'] = results['evalrextra2']/results['abh. Var']
    results['knn%'] = results['evalrknn2']/results['abh. Var']
    results['gradient%'] = results['evalrgradient2']/results['abh. Var']
    results['bag%'] = results['evalrbag2']/results['abh. Var']

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

    #print(results['xgb%'].mean())
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
