import pandas as pd
import numpy as np
import random as rnd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

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
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor


def main():

    warnings.filterwarnings("ignore")
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    df = pd.read_csv('Airlines.csv', delimiter=',')
    del df['id']
    corr = correlationn(df)

    with open('Alle Projekte.textmate', 'w') as file:
        file.write(str(df) + '\n')

    dataset = df_preprocessing(df)
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    Trainingsvolumen = 10000
    Testvolumen = 2000
    X_testt = (dataset.head(Trainingsvolumen+Testvolumen)).tail(Testvolumen)
    dataset = dataset.head(Trainingsvolumen)

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
        data = dataset
        results = predictions(data, i, X_testt)
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
    highrate.to_csv('Höchste Bewertungen.csv')


def df_preprocessing(df):

    print('Abschlusswahrscheinlichkeit wird berechnet...')

    X_cat = df[['Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek']]
    X_num = df.drop(['Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek', 'Delay'], axis=1)
    X_cat = pd.get_dummies(X_cat, drop_first=True)
    scaler = StandardScaler()
    scaler.fit(X_num)
    X_scaled = scaler.transform(X_num)
    X_scaled = pd.DataFrame(X_scaled, index=X_num.index, columns=X_num.columns)
    df = pd.concat([df['Delay'], X_scaled, X_cat], axis=1)

    return df


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


def correlationn(df):
    corr_mlr = df.corr(method='pearson')
    plt.figure(figsize=(20, 6))
    sns.heatmap(corr_mlr, annot=True, cmap='Blues')
    plt.title('Correlation matrix')
    plt.show()

    return corr_mlr


def predictions(data, i, X_testt):
    rlog = []; rsvc = []; rknn = []; rgaussian = []; rperception = []; rlinear_svc = []; rsgd = []; rdecision_tree = []; rrandom_forest = []; rxgb = []; rbag = []; rada = []
    rxgb2 = []; rdecision_tree2 = []; rrandom_forest2 = []; rada2 = []; rextra2 = []; rknn2 = []; rcat2 = []; rgradient2 = []; rbag2 = []

    df_train = data.head(i)
    Y_train = df_train['Delay']#.astype('int')
    del df_train['Delay']
    X_train = df_train#.astype('int')
    df_test = X_testt
    Y_test = df_test['Delay']
    del df_test['Delay']
    X_test = df_test#.astype('int')


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
    gradient2 = GradientBoostingRegressor()
    gradient2.fit(X_train.values, Y_train.values)
    bag2 = BaggingRegressor()
    bag2.fit(X_train.values, Y_train.values)


    for l in range(0, len(X_test)):
        z = (l + 1) / len(X_test)
        print('Fortschritt: ' + str(round(z * 100, 1)) + '%')
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
        prediction_gradient2 = gradient2.predict([values])
        rgradient2.append(round(prediction_gradient2[0], 3))
        prediction_bag2 = bag2.predict([values])
        rbag2.append(round(prediction_bag2[0], 3))


    df_testresults = pd.DataFrame()
    df_testresults['Delay'] = Y_test
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
    df_testresults['rgradient2'] = rgradient2
    df_testresults['rbag2'] = rbag2

    return df_testresults


def accuracyss(results,acclog,accsvc,accknn,accgaussian,accperception,acclinear_svc,accsgd,accdecision_tree,accrandom_forest,accxgb,accbag,accada):
    df_predictions = pd.DataFrame()

    y_test = results['Delay']
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

    y_test = results['Delay']
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

    y_test = results['Delay']
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
    y_test = results['Delay']
    y_pred = results['xgb'].astype(int)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(cnf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion matrix: XG Boost')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()


def confusionmatrix2(results):
    y_test = results['Delay']
    y_pred = results['gaussian'].astype(int)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(cnf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion matrix: Guaussian')
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
