import pandas as pd
import numpy as np
import random as rnd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

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


def main():

    warnings.filterwarnings("ignore")
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    df = pd.read_csv('lr0.csv', delimiter=';')
    del df['Einmalzahlung'], df['Rente'], df['PLZ'], df['Grundstuecksgroesse'], df['ID']
    with open('Alle Projekte.textmate', 'w') as file:
        file.write(str(df) + '\n')

    dataset = df_preprocessing(df)
    X_testt = dataset.loc[dataset['Erstellungsjahr'] == 2020]
    Projektnummer = X_testt['Projektnummer']
    del X_testt['Projektnummer'], dataset['Projektnummer']
    dataset = dataset.loc[dataset['Erstellungsjahr'] != 2020]

    del dataset['Erstellungsjahr'], X_testt['Erstellungsjahr']
    with open('Testset.textmate', 'w') as file:
        file.write(str(X_testt) + '\n')
    with open('Trainingset.textmate', 'w') as file:
        file.write(str(dataset) + '\n')

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

        highrate = highrated(results)

        df_predictions = accuracyss(results,acclog,accsvc,accknn,accgaussian,accperception,acclinear_svc,accsgd,accdecision_tree,accrandom_forest,accxgb,accbag,accada)
        precisions = precisionss(df_predictions, results, preclog,precsvc,precknn,precgaussian,precperception,preclinear_svc,precsgd,precdecision_tree,precrandom_forest,precxgb,precbag,precada)
        df_prediction = recallss(df_predictions, results, reclog,recsvc,recknn,recgaussian,recperception,reclinear_svc,recsgd,recdecision_tree,recrandom_forest,recxgb,recbag,recada)
    df_predictionss = df_prediction
    with open('df_predictionss.textmate', 'w') as file:
        file.write(str(df_predictionss) + '\n')
    predictions_stats = df_stats2(df_predictionss)
    print(predictions_stats)
    corr = correlation2(results)
    confusion = confusionmatrix(results)
    confusion = confusionmatrix2(results)
    highrate.to_csv('Höchste Bewertungen 2020.csv')


def df_preprocessing(df):

    print('Abschlusswahrscheinlichkeit wird berechnet...')

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
    Y_train = df_train['Status'].astype('int')
    del df_train['Status']
    X_train = df_train.astype('int')
    df_test = X_testt
    Y_test = df_test['Status']
    del df_test['Status']
    X_test = df_test.astype('int')


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
    xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)
    xgb.fit(X_train.values, Y_train.values)
    bag = BaggingClassifier()
    bag.fit(X_train.values, Y_train.values)
    ada = AdaBoostClassifier()
    ada.fit(X_train.values, Y_train.values)

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
        prediction_log = logreg.predict_proba([values])
        if prediction_log[0][1] > 0.15:
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
        prediction_xgb = xgb.predict([values])
        rxgb.append((prediction_xgb[0]))
        prediction_bag = bag.predict([values])
        rbag.append((prediction_bag[0]))
        prediction_ada = ada.predict([values])
        rada.append((prediction_ada[0]))

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
    df_testresults['Status'] = Y_test
    df_testresults['log'] = rlog
    df_testresults['svc'] = rsvc
    df_testresults['knn'] = rknn
    df_testresults['gaussian'] = rgaussian
    df_testresults['perception'] = rperception
    df_testresults['linear_svc'] = rlinear_svc
    df_testresults['sgd'] = rsgd
    df_testresults['decision_tree'] = rdecision_tree
    df_testresults['random_forest'] = rrandom_forest
    df_testresults['xgb'] = rxgb
    df_testresults['bag'] = rbag
    df_testresults['ada'] = rada

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


def accuracyss(results,acclog,accsvc,accknn,accgaussian,accperception,acclinear_svc,accsgd,accdecision_tree,accrandom_forest,accxgb,accbag,accada):
    df_predictions = pd.DataFrame()

    y_test = results['Status']
    y_pred1 = results['log']
    y_pred2 = results['svc']
    y_pred3 = results['knn']
    y_pred4 = results['gaussian']
    y_pred5 = results['perception']
    y_pred6 = results['linear_svc']
    y_pred7 = results['sgd']
    y_pred8 = results['decision_tree']
    y_pred9 = results['random_forest']
    y_pred10 = results['xgb']
    y_pred11 = results['bag']
    y_pred12 = results['ada']

    acclog.append(metrics.accuracy_score(y_test, y_pred1))
    accsvc.append(metrics.accuracy_score(y_test, y_pred2))
    accknn.append(metrics.accuracy_score(y_test, y_pred3))
    accgaussian.append(metrics.accuracy_score(y_test, y_pred4))
    accperception.append(metrics.accuracy_score(y_test, y_pred5))
    acclinear_svc.append(metrics.accuracy_score(y_test, y_pred6))
    accsgd.append(metrics.accuracy_score(y_test, y_pred7))
    accdecision_tree.append(metrics.accuracy_score(y_test, y_pred8))
    accrandom_forest.append(metrics.accuracy_score(y_test, y_pred9))
    accxgb.append(metrics.accuracy_score(y_test, y_pred10))
    accbag.append(metrics.accuracy_score(y_test, y_pred11))
    accada.append(metrics.accuracy_score(y_test, y_pred12))

    df_predictions['acclog'] = acclog
    df_predictions['accsvc'] = accsvc
    df_predictions['accknn'] = accknn
    df_predictions['accgaussian'] = accgaussian
    df_predictions['accperception'] = accperception
    df_predictions['acclinear_svc'] = acclinear_svc
    df_predictions['accsgd'] = accsgd
    df_predictions['accdecision_tree'] = accdecision_tree
    df_predictions['accrandom_forest'] = accrandom_forest
    df_predictions['accxgb'] = accxgb
    df_predictions['accbag'] = accbag
    df_predictions['accada'] = accada

    return df_predictions


def precisionss(df_predictions, results, preclog, precsvc, precknn, precgaussian, precperception, preclinear_svc, precsgd, precdecision_tree, precrandom_forest, precxgb, precbag, precada):

    y_test = results['Status']
    y_pred1 = results['log']
    y_pred2 = results['svc']
    y_pred3 = results['knn']
    y_pred4 = results['gaussian']
    y_pred5 = results['perception']
    y_pred6 = results['linear_svc']
    y_pred7 = results['sgd']
    y_pred8 = results['decision_tree']
    y_pred9 = results['random_forest']
    y_pred10 = results['xgb']
    y_pred11 = results['bag']
    y_pred12 = results['ada']

    preclog.append(metrics.precision_score(y_test, y_pred1))
    precsvc.append(metrics.precision_score(y_test, y_pred2))
    precknn.append(metrics.precision_score(y_test, y_pred3))
    precgaussian.append(metrics.precision_score(y_test, y_pred4))
    precperception.append(metrics.precision_score(y_test, y_pred5))
    preclinear_svc.append(metrics.precision_score(y_test, y_pred6))
    precsgd.append(metrics.precision_score(y_test, y_pred7))
    precdecision_tree.append(metrics.precision_score(y_test, y_pred8))
    precrandom_forest.append(metrics.precision_score(y_test, y_pred9))
    precxgb.append(metrics.precision_score(y_test, y_pred10))
    precbag.append(metrics.precision_score(y_test, y_pred11))
    precada.append(metrics.precision_score(y_test, y_pred12))

    df_predictions['preclog'] = preclog
    df_predictions['precsvc'] = precsvc
    df_predictions['precknn'] = precknn
    df_predictions['precgaussian'] = precgaussian
    df_predictions['precperception'] = precperception
    df_predictions['preclinear_svc'] = preclinear_svc
    df_predictions['precsgd'] = precsgd
    df_predictions['precdecision_tree'] = precdecision_tree
    df_predictions['precrandom_forest'] = precrandom_forest
    df_predictions['precxgb'] = precxgb
    df_predictions['precbag'] = precbag
    df_predictions['precada'] = precada

    return df_predictions


def recallss(df_predictions, results, reclog, recsvc, recknn, recgaussian, recperception, reclinear_svc, recsgd,recdecision_tree, recrandom_forest, recxgb, recbag, recada):

    y_test = results['Status']
    y_pred1 = results['log']
    y_pred2 = results['svc']
    y_pred3 = results['knn']
    y_pred4 = results['gaussian']
    y_pred5 = results['perception']
    y_pred6 = results['linear_svc']
    y_pred7 = results['sgd']
    y_pred8 = results['decision_tree']
    y_pred9 = results['random_forest']
    y_pred10 = results['xgb']
    y_pred11 = results['bag']
    y_pred12 = results['ada']

    reclog.append(metrics.recall_score(y_test, y_pred1))
    recsvc.append(metrics.recall_score(y_test, y_pred2))
    recknn.append(metrics.recall_score(y_test, y_pred3))
    recgaussian.append(metrics.recall_score(y_test, y_pred4))
    recperception.append(metrics.recall_score(y_test, y_pred5))
    reclinear_svc.append(metrics.recall_score(y_test, y_pred6))
    recsgd.append(metrics.recall_score(y_test, y_pred7))
    recdecision_tree.append(metrics.recall_score(y_test, y_pred8))
    recrandom_forest.append(metrics.recall_score(y_test, y_pred9))
    recxgb.append(metrics.recall_score(y_test, y_pred10))
    recbag.append(metrics.recall_score(y_test, y_pred11))
    recada.append(metrics.recall_score(y_test, y_pred12))

    df_predictions['reclog'] = reclog
    df_predictions['recsvc'] = recsvc
    df_predictions['recknn'] = recknn
    df_predictions['recgaussian'] = recgaussian
    df_predictions['recperception'] = recperception
    df_predictions['reclinear_svc'] = reclinear_svc
    df_predictions['recsgd'] = recsgd
    df_predictions['recdecision_tree'] = recdecision_tree
    df_predictions['recrandom_forest'] = recrandom_forest
    df_predictions['recxgb'] = recxgb
    df_predictions['recbag'] = recbag
    df_predictions['recada'] = recada

    return df_predictions


def df_stats2(df_predictionss):
    df_statistics_predictions = pd.DataFrame()
    df_statistics_predictions['Mean'] = round(df_predictionss.mean(), 3)
    df_statistics_predictions['Median'] = round(df_predictionss.median(), 3)
    df_statistics_predictions['Max'] = round(df_predictionss.max(), 3)
    df_statistics_predictions['Min'] = round(df_predictionss.min(), 3)
    df_statistics_predictions['Stand. dev.'] = df_predictionss.std()

    return df_statistics_predictions


def correlation2(results):
    corr_mlr = results.corr(method='pearson')
    plt.figure(figsize=(20, 6))
    sns.heatmap(corr_mlr, annot=True, cmap='Blues')
    plt.title('Correlation matrix')
    plt.show()

    return corr_mlr


def confusionmatrix(results):
    y_test = results['Status']
    y_pred = results['xgb'].astype(int)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(cnf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion matrix: XG Boost')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()


def confusionmatrix2(results):
    y_test = results['Status']
    y_pred = results['knn'].astype(int)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(cnf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion matrix: K-Nearest')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()


def highrated(results):
    highrated = pd.DataFrame()
    highrated = results
    highrated = highrated.loc[highrated['xgb2'] > -2000]
    #highrated = highrated.loc[highrated['Status'] == 0]
    highrated = highrated.sort_values(by=['xgb2'], ascending=False)
    with open('highrated.textmate', 'w') as file:
        file.write(str(highrated) + '\n')

    return highrated


main()