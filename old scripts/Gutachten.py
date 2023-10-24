import pandas as pd
import numpy as np
import random as rnd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# machine learning
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor


def main():
    warnings.filterwarnings("ignore")
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    df = pd.read_csv('lr0.csv', delimiter=';')
    del df['Einmalzahlung'], df['Rente'], df['PLZ'], df['Grundstuecksgroesse'], df['ID']
    df['Gutachten'] = df['Gutacherwert']
    with open('Alle Projekte.textmate', 'w') as file:
        file.write(str(df) + '\n')
    print(df['Gutacherwert'].count())

    dataset = df_preprocessing(df)
    #del dataset["Status"], dataset["Erstellungsjahr"], dataset["Herkunft"], dataset["Geschlecht"], dataset["Alter"], dataset["Kinder"], dataset["Familienstand"],dataset["Alter2"]

    '''X_testt = dataset.loc[pd.isnull(dataset["Gutachten"])]
    Projektnummer = X_testt['Projektnummer']
    del X_testt['Projektnummer'], dataset['Projektnummer'], X_testt['Immobilienwert_intern'], dataset['Immobilienwert_intern']
    dataset = dataset.loc[pd.notnull(dataset["Gutachten"])]
    del X_testt['Gutachten'], dataset['Gutachten']'''

    dataset = dataset.sample(frac=1).reset_index(drop=True)
    X_testt = dataset.loc[pd.notnull(dataset["Gutachten"])]
    X_testt = X_testt.tail(218)
    Projektnummer = X_testt['Projektnummer']
    del X_testt['Projektnummer'], dataset['Projektnummer'], X_testt['Immobilienwert_intern'], dataset[
        'Immobilienwert_intern']
    dataset = dataset.loc[pd.notnull(dataset["Gutachten"])]
    dataset = dataset.head(1300)
    del X_testt['Gutachten'], dataset['Gutachten']


    with open('Trainingset.textmate', 'w') as file:
        file.write(str(dataset) + '\n')
    with open('Testset.textmate', 'w') as file:
        file.write(str(X_testt) + '\n')
    df_statistics = df_stats(dataset)
    #corr = correlation(dataset)
    with open('Infos Trainingset.textmate', 'w') as file:
        file.write(str(df_statistics) + '\n')

    i = int(len(dataset))
    acclog = []; accsvc = []; accknn = []; accgaussian = []; accperception = []; acclinear_svc = []; accsgd = []; accdecision_tree = []; accrandom_forest = []; accxgb = []; accbag = []; accada = []
    preclog = []; precsvc = []; precknn = []; precgaussian = []; precperception = []; preclinear_svc = []; precsgd = []; precdecision_tree = []; precrandom_forest = []; precxgb = []; precbag = []; precada = []
    reclog = []; recsvc = []; recknn = []; recgaussian = []; recperception = []; reclinear_svc = []; recsgd = []; recdecision_tree = []; recrandom_forest = []; recxgb = []; recbag = []; recada = []
    df_prediction = pd.DataFrame()
    results = pd.DataFrame()
    highrate = pd.DataFrame()

    for x in range(0, 1):
        data = dataset.sample(frac=1).reset_index(drop=True)
        results = predictions(data, i, X_testt, Projektnummer)
        with open('df_results.textmate', 'w') as file:
            file.write(str(results) + '\n')
    mse = rmse(results)
    corr = correlation2(results)



def df_preprocessing(df):

    print('Gutachterwerte werden berechnet...')

    # Status
    df['Status'] = df['Status'].replace('wird gelöscht', 0)
    df['Status'] = df['Status'].replace('Rückabwicklung', 0)
    df['Status'] = df['Status'].replace('Fragebogen liegt vor', 0)
    df['Status'] = df['Status'].replace('Gutachten beauftragt', 0)
    df['Status'] = df['Status'].replace('Gutachten liegt vor', 0)
    df['Status'] = df['Status'].replace('Gutachtenauftrag in Vorbereitung', 0)
    df['Status'] = df['Status'].replace('In Besitz der DLAG', 1)
    df['Status'] = df['Status'].replace('Kein Interesse mehr', 0)
    df['Status'] = df['Status'].replace('Kunde hat abgelehnt', 0)
    df['Status'] = df['Status'].replace('Kunde hat abgelehnt nach verb. Angebot', 0)
    df['Status'] = df['Status'].replace('Kunde wurde abgelehnt', 0)
    df['Status'] = df['Status'].replace('Neukunde', 0)
    df['Status'] = df['Status'].replace('Notartermin', 0)
    df['Status'] = df['Status'].replace('Notarvertrag angefordert', 0)
    df['Status'] = df['Status'].replace('Notarvertrag versandt', 0)
    df['Status'] = df['Status'].replace('unverbindliches Angebot versandt', 0)
    df['Status'] = df['Status'].replace('verbindliches Angebot erstellt', 0)
    df['Status'] = df['Status'].replace('Verkauft', 1)
    df['Status'] = df['Status'].replace('Vermittler zugeordnet', 0)
    df['Status'] = df['Status'].replace('verschoben', 0)
    df['Status'] = df['Status'].replace('Vertragsannahme', 1)
    # Herkunft
    df['Herkunft'] = df['Herkunft'].fillna(df['Herkunft'].mean())
    # Kanal
    df['Kanal'] = df['Kanal'].fillna('Unbekannt')
    # Geschlecht
    df['Geschlecht'] = df['Geschlecht'].replace('Herr', 2)
    df['Geschlecht'] = df['Geschlecht'].replace('Frau', 1)
    df['Geschlecht'] = df['Geschlecht'].fillna(df['Geschlecht'].mean())
    # Alter
    #print(df[["Kanal", "Alter"]].groupby(['Kanal'], as_index=False).mean().sort_values(by='Alter',ascending=False))
    df2 = df.loc[df['Kanal'] == 'Unbekannt']
    df2['Alter'] = df2['Alter'].fillna(df2['Alter'].mean())
    df3 = df.loc[df['Kanal'] == 'Verband Pflegehilfe']
    df3['Alter'] = df3['Alter'].fillna(df3['Alter'].mean())
    df4 = df.loc[df['Kanal'] == 'Flyer']
    df4['Alter'] = df4['Alter'].fillna(df4['Alter'].mean())
    df5 = df.loc[df['Kanal'] == 'Zeitung']
    df5['Alter'] = df5['Alter'].fillna(df5['Alter'].mean())
    df6 = df.loc[df['Kanal'] == 'Empfehlung']
    df6['Alter'] = df6['Alter'].fillna(df6['Alter'].mean())
    df7 = df.loc[df['Kanal'] == 'Radio']
    df7['Alter'] = df7['Alter'].fillna(df7['Alter'].mean())
    df8 = df.loc[df['Kanal'] == 'Fernsehen']
    df8['Alter'] = df8['Alter'].fillna(df8['Alter'].mean())
    df9 = df.loc[df['Kanal'] == 'Event']
    df9['Alter'] = df9['Alter'].fillna(df9['Alter'].mean())
    df10 = df.loc[df['Kanal'] == 'Immoverkauf24']
    df10['Alter'] = df10['Alter'].fillna(df10['Alter'].mean())
    df11 = df.loc[df['Kanal'] == 'Social Media']
    df11['Alter'] = df11['Alter'].fillna(df11['Alter'].mean())
    df12 = df.loc[df['Kanal'] == 'Freunde werben Freunde']
    df12['Alter'] = df12['Alter'].fillna(df12['Alter'].mean())
    df13 = df.loc[df['Kanal'] == 'Internet']
    df13['Alter'] = df13['Alter'].fillna(df13['Alter'].mean())
    df14 = df.loc[df['Kanal'] == 'tipster']
    df14['Alter'] = df14['Alter'].fillna(df14['Alter'].mean())
    df15 = df.loc[df['Kanal'] == 'LBSI Nord-West']
    df15['Alter'] = df15['Alter'].fillna(df15['Alter'].mean())
    df16 = df.loc[df['Kanal'] == 'Andere']
    df16['Alter'] = df16['Alter'].fillna(df16['Alter'].mean())
    dfb = df2.append(df3, ignore_index=True)
    dfc = dfb.append(df4, ignore_index=True)
    dfd = dfc.append(df5, ignore_index=True)
    dfe = dfd.append(df6, ignore_index=True)
    dff = dfe.append(df7, ignore_index=True)
    dfg = dff.append(df8, ignore_index=True)
    dfh = dfg.append(df9, ignore_index=True)
    dfi = dfh.append(df10, ignore_index=True)
    dfj = dfi.append(df11, ignore_index=True)
    dfk = dfj.append(df12, ignore_index=True)
    dfl = dfk.append(df13, ignore_index=True)
    dfm = dfl.append(df14, ignore_index=True)
    dfn = dfm.append(df15, ignore_index=True)
    df1 = dfn.append(df16, ignore_index=True)
    # Kanal
    df1['Kanal'] = df1['Kanal'].fillna(value=0)
    df1['Kanal'] = df1['Kanal'].replace('Unbekannt', 0)
    df1['Kanal'] = df1['Kanal'].replace('Verband Pflegehilfe', 1)
    df1['Kanal'] = df1['Kanal'].replace('Flyer', 2)
    df1['Kanal'] = df1['Kanal'].replace('Zeitung', 3)
    df1['Kanal'] = df1['Kanal'].replace('Empfehlung', 4)
    df1['Kanal'] = df1['Kanal'].replace('Radio', 5)
    df1['Kanal'] = df1['Kanal'].replace('Fernsehen', 6)
    df1['Kanal'] = df1['Kanal'].replace('Event', 7)
    df1['Kanal'] = df1['Kanal'].replace('Immoverkauf24', 8)
    df1['Kanal'] = df1['Kanal'].replace('Social Media', 9)
    df1['Kanal'] = df1['Kanal'].replace('Freunde werben Freunde', 10)
    df1['Kanal'] = df1['Kanal'].replace('Internet', 11)
    df1['Kanal'] = df1['Kanal'].replace('tipster', 12)
    df1['Kanal'] = df1['Kanal'].replace('LBSI Nord-West', 13)
    df1['Kanal'] = df1['Kanal'].replace('Andere', 13)
    # Kinder
    df1.loc[df1['Kinder'] >= -3000, 'Kinder'] = 1
    df1['Kinder'] = df1['Kinder'].fillna(value=0)
    # Alter2
    df1.loc[df1['Alter2'] >= -3000, 'Alter2'] = 1
    df1['Alter2'] = df1['Alter2'].fillna(value=0)
    # Familienstand
    df2 = df1.loc[df1['Alter2'] == 0]
    df2['Familienstand'] = df2['Familienstand'].fillna(df2['Familienstand'].mean())
    df3 = df1.loc[df1['Alter2'] == 1]
    df3['Familienstand'] = df3['Familienstand'].fillna(df3['Familienstand'].mean())
    df = df2.append(df3, ignore_index=True)
    # Bundesland
    df['Bundesland'] = df['Bundesland'].fillna(value=9)
    df['Bundesland'] = df['Bundesland'].replace('Niedersachen', 9)
    df['Bundesland'] = df['Bundesland'].replace('Thüringen', 7)
    df['Bundesland'] = df['Bundesland'].replace('Schleswig-Holstein', 10)
    df['Bundesland'] = df['Bundesland'].replace('Sachsen-Anhalt', 3)
    df['Bundesland'] = df['Bundesland'].replace('Sachsen', 2)
    df['Bundesland'] = df['Bundesland'].replace('Saarland', 6)
    df['Bundesland'] = df['Bundesland'].replace('Rheinland-Pfalz', 11)
    df['Bundesland'] = df['Bundesland'].replace('Nordrhein-Westfalen', 15)
    df['Bundesland'] = df['Bundesland'].replace('Niedersachsen', 5)
    df['Bundesland'] = df['Bundesland'].replace('Mecklenburg-Vorpommern', 4)
    df['Bundesland'] = df['Bundesland'].replace('Hessen', 12)
    df['Bundesland'] = df['Bundesland'].replace('Hamburg', 17)
    df['Bundesland'] = df['Bundesland'].replace('Bremen', 8)
    df['Bundesland'] = df['Bundesland'].replace('Brandenburg', 1)
    df['Bundesland'] = df['Bundesland'].replace('Berlin', 16)
    df['Bundesland'] = df['Bundesland'].replace('Bayern', 14)
    df['Bundesland'] = df['Bundesland'].replace('Baden-Württemberg', 13)
    # Objektart
    df['Objektart'] = df['Objektart'].replace('Sonstiges', 1)
    df['Objektart'] = df['Objektart'].replace('Eigentumswohnung', 2)
    df['Objektart'] = df['Objektart'].replace('Reihenhaus', 3)
    df['Objektart'] = df['Objektart'].replace('Einfamilienhaus', 4)
    df['Objektart'] = df['Objektart'].replace('Doppelhaushälfte', 5)
    df['Objektart'] = df['Objektart'].replace('Zweifamilienhaus', 6)
    df['Objektart'] = df['Objektart'].replace('Mehrfamilienhaus', 7)
    df['Objektart'] = df['Objektart'].fillna(value=0)
    df['Objektart'] = df['Objektart'].replace('nan', 0)
    # Keller
    df['Keller'] = df['Keller'].fillna(1)
    # Wohnlage
    df['Wohnlage'] = df['Wohnlage'].fillna(df['Wohnlage'].mean())
    # Objektzustand
    df['Objektzustand'] = df['Objektzustand'].fillna(df['Objektzustand'].mean())
    # Wohnhaft
    df['Wohnhaft'] = df['Wohnhaft'].fillna(df['Wohnhaft'].mean())
    # Heizung
    df2 = df.loc[df['Wohnhaft'] < 1980]
    df2['Heizung'] = df2['Heizung'].fillna(df2['Heizung'].mean())
    df3 = df.loc[df['Wohnhaft'] > 1979]
    df3['Heizung'] = df3['Heizung'].fillna(df3['Heizung'].mean())
    df = df2.append(df3, ignore_index=True)
    # Wohnflaeche
    df["Wohnflaeche"] = [float(str(i).replace(",", ".")) for i in df["Wohnflaeche"]]
    df2 = df.loc[df['Objektart'] == 0]
    df2['Wohnflaeche'] = df2['Wohnflaeche'].fillna(df2['Wohnflaeche'].mean())
    df3 = df.loc[df['Objektart'] == 1]
    df3['Wohnflaeche'] = df3['Wohnflaeche'].fillna(df3['Wohnflaeche'].mean())
    df4 = df.loc[df['Objektart'] == 2]
    df4['Wohnflaeche'] = df4['Wohnflaeche'].fillna(df4['Wohnflaeche'].mean())
    df5 = df.loc[df['Objektart'] == 3]
    df5['Wohnflaeche'] = df5['Wohnflaeche'].fillna(df5['Wohnflaeche'].mean())
    df6 = df.loc[df['Objektart'] == 4]
    df6['Wohnflaeche'] = df6['Wohnflaeche'].fillna(df6['Wohnflaeche'].mean())
    df7 = df.loc[df['Objektart'] == 5]
    df7['Wohnflaeche'] = df7['Wohnflaeche'].fillna(df7['Wohnflaeche'].mean())
    df8 = df.loc[df['Objektart'] == 6]
    df8['Wohnflaeche'] = df8['Wohnflaeche'].fillna(df8['Wohnflaeche'].mean())
    df9 = df.loc[df['Objektart'] == 7]
    df9['Wohnflaeche'] = df9['Wohnflaeche'].fillna(df9['Wohnflaeche'].mean())
    dfb = df2.append(df3, ignore_index=True)
    dfc = dfb.append(df4, ignore_index=True)
    dfd = dfc.append(df5, ignore_index=True)
    dfe = dfd.append(df6, ignore_index=True)
    dff = dfe.append(df7, ignore_index=True)
    dfg = dff.append(df8, ignore_index=True)
    df = dfg.append(df9, ignore_index=True)
    # Garagen
    df.loc[df['Garagen'] >= 1, 'Garagen'] = 1
    df['Garagen'] = df['Garagen'].fillna(value=0)
    # Stellplätze
    df.loc[df['Stellplätze'] >= 1, 'Stellplätze'] = 1
    df['Stellplätze'] = df['Stellplätze'].fillna(value=0)
    # Immobilienwert_intern
    df['Immobilienwert_intern'] = df['Immobilienwert_intern'].replace(0, None)
    df3 = df.loc[df['Wohnlage'] <= 1]
    df3['Immobilienwert_intern'] = df3['Immobilienwert_intern'].fillna(df3['Immobilienwert_intern'].mean())
    df4 = df.loc[df['Wohnlage'] == 2]
    df4['Immobilienwert_intern'] = df4['Immobilienwert_intern'].fillna(df4['Immobilienwert_intern'].mean())
    df6 = df.loc[df['Wohnlage'] > 2]
    df6['Immobilienwert_intern'] = df6['Immobilienwert_intern'].fillna(df6['Immobilienwert_intern'].mean())
    dfc = df3.append(df4, ignore_index=True)
    dfx = dfc.append(df6, ignore_index=True)
    # Gutacherwert
    dfx['Gutacherwert'] = dfx['Gutacherwert'].replace(0, None)
    df3 = dfx.loc[dfx['Wohnlage'] <= 1]
    df3['Gutacherwert'] = df3['Gutacherwert'].fillna(df3['Gutacherwert'].mean())
    df4 = dfx.loc[dfx['Wohnlage'] == 2]
    df4['Gutacherwert'] = df4['Gutacherwert'].fillna(df4['Gutacherwert'].mean())
    df6 = dfx.loc[dfx['Wohnlage'] > 2]
    df6['Gutacherwert'] = df6['Gutacherwert'].fillna(df6['Gutacherwert'].mean())
    dfc = df3.append(df4, ignore_index=True)
    dfy = dfc.append(df6, ignore_index=True)

    return dfy


def df_stats(dataset):
    df_statistics = pd.DataFrame()
    df_statistics['mean'] = dataset.mean()
    df_statistics['median'] = dataset.median()
    df_statistics['max'] = dataset.max()
    df_statistics['min'] = dataset.min()
    df_statistics['count'] = dataset.count()
    df_statistics['std'] = dataset.std()

    return df_statistics


def correlation(dataset):
    corr_mlr = dataset.corr(method='pearson')
    plt.figure(figsize=(20, 6))
    sns.heatmap(corr_mlr, annot=True, cmap='Blues')
    plt.title('Correlation matrix')
    plt.show()

    return corr_mlr


def predictions(data, i, X_testt, Projektnummer):
    rlog = []; rsvc = []; rknn = []; rgaussian = []; rperception = []; rlinear_svc = []; rsgd = []; rdecision_tree = []; rrandom_forest = []; rxgb = []; rbag = []; rada = []
    rxgb2 = []; rdecision_tree2 = []; rrandom_forest2 = []; rada2 = []; rextra2 = []; rknn2 = []; rcat2 = []; rgradient2 = []; rbag2 = []

    df_train = data.head(i)
    Y_train = df_train['Gutacherwert'].astype('int')
    del df_train['Gutacherwert']
    X_train = df_train.astype('int')
    df_test = X_testt
    Y_test = df_test['Gutacherwert']
    del df_test['Gutacherwert']
    X_test = df_test.astype('int')


    logreg = LogisticRegression()
    logreg.fit(X_train.values, Y_train.values)

    # create an xgboost regression model
    xgb2 = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    xgb2.fit(X_train.values, Y_train.values)
    decision_tree2 = DecisionTreeRegressor()
    decision_tree2.fit(X_train.values, Y_train.values)
    random_forest2 = RandomForestRegressor()
    random_forest2.fit(X_train.values, Y_train.values)
    ada2 = AdaBoostRegressor()
    ada2.fit(X_train.values, Y_train.values)
    extra2 = ExtraTreesRegressor()
    extra2.fit(X_train.values, Y_train.values)
    knn2 = KNeighborsRegressor()
    knn2.fit(X_train.values, Y_train.values)
    cat2 = CatBoostRegressor()
    cat2.fit(X_train.values, Y_train.values)
    gradient2 = GradientBoostingRegressor()
    gradient2.fit(X_train.values, Y_train.values)
    bag2 = BaggingRegressor()
    bag2.fit(X_train.values, Y_train.values)


    for l in range(0, len(X_test)):
        values = X_test.values[l]

        prediction_xgb2 = xgb2.predict([values])
        rxgb2.append(round(prediction_xgb2[0], 3))
        prediction_decision_tree2 = decision_tree2.predict([values])
        rdecision_tree2.append(round(prediction_decision_tree2[0], 3))
        prediction_random_forest2 = random_forest2.predict([values])
        rrandom_forest2.append(round(prediction_random_forest2[0], 3))
        prediction_ada2 = ada2.predict([values])
        rada2.append(round(prediction_ada2[0], 3))
        prediction_extra2 = extra2.predict([values])
        rextra2.append(round(prediction_extra2[0], 3))
        prediction_knn2 = knn2.predict([values])
        rknn2.append(round(prediction_knn2[0], 3))
        prediction_cat2 = cat2.predict([values])
        rcat2.append(round(prediction_cat2[0], 3))
        prediction_gradient2 = gradient2.predict([values])
        rgradient2.append(round(prediction_gradient2[0], 3))
        prediction_bag2 = bag2.predict([values])
        rbag2.append(round(prediction_bag2[0], 3))


    df_testresults = pd.DataFrame()
    df_testresults['Gutacherwert'] = Y_test

    df_testresults['xgb2'] = rxgb2
    df_testresults['rdecision_tree2'] = rdecision_tree2
    df_testresults['rrandom_forest2'] = rrandom_forest2
    df_testresults['rada2'] = rada2
    df_testresults['rextra2'] = rextra2
    df_testresults['rextra2'] = rextra2
    df_testresults['rknn2'] = rknn2
    df_testresults['rcat2'] = rcat2
    df_testresults['rgradient2'] = rgradient2
    df_testresults['rbag2'] = rbag2

    df_testresults['Projektnummer'] = Projektnummer

    return df_testresults


def correlation2(results):
    corr_mlr = results.corr(method='pearson')
    plt.figure(figsize=(20, 6))
    sns.heatmap(corr_mlr, annot=True, cmap='Blues')
    plt.title('Correlation matrix')
    plt.show()

    return corr_mlr


def rmse(results):
    print(np.sqrt(mean_squared_error(results['Gutacherwert'], results['xgb2'])))
    print(np.sqrt(mean_squared_error(results['Gutacherwert'], results['rdecision_tree2'])))
    print(np.sqrt(mean_squared_error(results['Gutacherwert'], results['rrandom_forest2'])))
    print(np.sqrt(mean_squared_error(results['Gutacherwert'], results['rada2'])))
    print(np.sqrt(mean_squared_error(results['Gutacherwert'], results['rextra2'])))
    print(np.sqrt(mean_squared_error(results['Gutacherwert'], results['rknn2'])))
    print(np.sqrt(mean_squared_error(results['Gutacherwert'], results['rcat2'])))
    print(np.sqrt(mean_squared_error(results['Gutacherwert'], results['rgradient2'])))
    print(np.sqrt(mean_squared_error(results['Gutacherwert'], results['rbag2'])))


main()

