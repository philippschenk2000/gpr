import math
from scipy.stats import ttest_ind
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
import scipy.stats as stats
import statsmodels.formula.api as sm
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import datasets


def main():

    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    df = pd.read_csv('index.csv', delimiter=',')
    df_statistics = df_stats(df)
    with open('df_statistics.textmate', 'w') as file:
        file.write(str(df_statistics) + '\n')
    del df['Purpose'], df['Guarantors'], df['Duration in Current address'], df['Type of apartment'], df['No of Credits at this Bank'], df['Occupation'], df['No of dependents'], df['Telephone']
    df_rmse = pd.DataFrame()
    df_results = pd.DataFrame()
    df_predictions = pd.DataFrame()
    training_rows = []
    rmse_result = []
    rmse_result1 = []
    rmse_result2 = []
    rmse_result3 = []


    '''Parameters to choose for Calculation'''
    trainingsmatrix_rows_at_end = 600       # everything except rmse-values are calculated with this amount of rows
    trainingsmatrix_rows_at_start = 599     # just for rmse-analysis needed, usually trainingsmatrix_rows_at_end - 1
    step = 1                                # steps between start and end, usually = 1
    length = 10000                            # times of mixing df_gesamt and times of prediction before the dataset gets x rows(=step) longer
    '''Parameters to choose for Calculation'''


    start = trainingsmatrix_rows_at_start
    end = trainingsmatrix_rows_at_end
    df1 = df_ges(df)

    for i in range(start, end, step):
        rmse_res = []
        rmse_res1 = []
        rmse_res2 = []
        rmse_res3 = []
        accuracy = []
        precision = []
        recall = []
        accuracy2 = []
        precision2 = []
        recall2 = []
        training_rows.append(start)
        start += 1*step
        df1 = df_ges(df)

        for x in range(0, length):
            df_gesamt = df1.sample(frac=1).reset_index(drop=True)
            with open('df_gesamt.textmate', 'w') as file:
                file.write(str(df_gesamt) + '\n')

            ''' for testing without age and sex make the next line readable and change xvalues in training and training2'''
            del df_gesamt['Credit_Amount']
            del df_gesamt['Age']
            del df_gesamt['SexandMarital_Status']
            del df_gesamt['Foreign Worker']
            '''end'''

            #corr = correlation(df_gesamt)
            mult_lin_reg = training(df_gesamt, i)
            mult_log_reg = training2(df_gesamt, i)
            df_results = testing(df_gesamt, i, mult_lin_reg, mult_log_reg)
            rrmse_lin = rmse_lin(df_results, rmse_res)
            #rrmse_og = rmse_log(df_results, rmse_res1)
            rrmse_bin_lin = rmse_bin_lin(df_results, rmse_res2)
            rrmse_bin_log = rmse_bin_log(df_results, rmse_res3)
            accura = acc(df_results)
            accuracy.append(accura)
            precis = prec(df_results)
            precision.append(precis)
            reca = rec(df_results)
            recall.append(reca)
            accurac = acc2(df_results)
            accuracy2.append(accurac)
            precisi = prec2(df_results)
            precision2.append(precisi)
            recal = rec2(df_results)
            recall2.append(recal)
            with open('df_results.textmate', 'w') as file:
                file.write(str(df_results) + '\n')

        rmse_result.append(sum(rmse_res)/length)
        #rmse_result1.append(sum(rmse_res1)/length)
        rmse_result2.append(sum(rmse_res2)/length)
        rmse_result3.append(sum(rmse_res3)/length)
        df_predictions['accuracy_lin_reg'] = accuracy
        df_predictions['precision_lin_reg'] = precision
        df_predictions['recall_lin_reg'] = recall
        df_predictions['accuracy_log_reg'] = accuracy2
        df_predictions['precision_log_reg'] = precision2
        df_predictions['recall_log_reg'] = recall2

    df_rmse['training_row'] = training_rows
    df_rmse['rmse_LINR'] = rmse_result
    #df_rmse['rmse_LOGR'] = rmse_result1
    df_rmse['rmse_BIN_LINR'] = rmse_result2
    df_rmse['rmse_BIN_LOGR'] = rmse_result3
    with open('df_rmse.textmate', 'w') as file:
        file.write(str(df_rmse) + '\n')
    with open('df_predictions.textmate', 'w') as file:
        file.write(str(df_predictions) + '\n')
    statistics_predictions = df_stats2(df_predictions)
    print(statistics_predictions)
    res = ttest_ind(df_predictions['accuracy_lin_reg'], df_predictions['accuracy_log_reg'])
    print(res)
    ress = ttest_ind(df_predictions['precision_lin_reg'], df_predictions['precision_log_reg'])
    print(ress)
    resss = ttest_ind(df_predictions['recall_lin_reg'], df_predictions['recall_log_reg'])
    print(resss)

    plotting_rmse = plottingrmse(training_rows, rmse_result, rmse_result1, rmse_result2, rmse_result3)
    confusion_matrix = confusionmatrix(df_results)
    confusion_matrix2 = confusionmatrix2(df_results)
    correlationsmatrix =correlationmatrix(df_results)
    roc_curve = roccurve(df_results)
    plt.show()


def df_ges(df):
    df_ges = pd.DataFrame()
    df_ges = df

    '''df_ges['Generation'] = np.where(df['Age'].between(18, 30), 1, df_ges['Age'])
    df_ges['Generation'] = np.where(df['Age'].between(30, 42), 2, df_ges['Generation'])
    df_ges['Generation'] = np.where(df['Age'].between(42, 80), 3, df_ges['Generation'])
    c14 = stats.linregress(df_ges['Generation'], df_ges['Creditability'])
    print("{:.6f}".format(float(c14[2] ** 2)))'''

    return df_ges


def df_stats(df):
    df_statistics = pd.DataFrame()
    df_statistics['mean'] = df.mean()
    df_statistics['median'] = df.median()
    df_statistics['max'] = df.max()
    df_statistics['min'] = df.min()
    df_statistics['count'] = df.count()
    df_statistics['std'] = df.std()

    a = []
    c1 = stats.linregress(df['Creditability'], df['Creditability'])
    #slope, intercept, r_value, p_value, std_err = stats.linregress(df['Duration_of_Credit_month'], df['Creditability'])
    c2 = stats.linregress(df['Account Balance'], df['Creditability'])
    c3 = stats.linregress(df['Duration_of_Credit_month'], df['Creditability'])
    c4 = stats.linregress(df['Payment Status of Prev. C.'], df['Creditability'])
    c5 = stats.linregress(df['Purpose'], df['Creditability'])
    c6 = stats.linregress(df['Credit_Amount'], df['Creditability'])
    c7 = stats.linregress(df['Value Savings/Stocks'], df['Creditability'])
    c8 = stats.linregress(df['Length of current employment'], df['Creditability'])
    c9 = stats.linregress(df['Instalment per cent'], df['Creditability'])
    c10 = stats.linregress(df['SexandMarital_Status'], df['Creditability'])
    c11 = stats.linregress(df['Guarantors'], df['Creditability'])
    c12 = stats.linregress(df['Duration in Current address'], df['Creditability'])
    c13 = stats.linregress(df['Most valuable asset'], df['Creditability'])
    c14 = stats.linregress(df['Age'], df['Creditability'])
    c15 = stats.linregress(df['Concurrent Credits'], df['Creditability'])
    c16 = stats.linregress(df['Type of apartment'], df['Creditability'])
    c17 = stats.linregress(df['No of Credits at this Bank'], df['Creditability'])
    c18 = stats.linregress(df['Occupation'], df['Creditability'])
    c19 = stats.linregress(df['No of dependents'], df['Creditability'])
    c20 = stats.linregress(df['Telephone'], df['Creditability'])
    c21 = stats.linregress(df['Foreign Worker'], df['Creditability'])
    p=2
    a = ["{:.6f}".format(float(c1[p]**2)), c2[p]**2, c3[p]**2, c4[p]**2, c5[p]**2, c6[p]**2, c7[p]**2, c8[p]**2, c9[p]**2, c10[p]**2, c11[p]**2, c12[p]**2, c13[p]**2, c14[p]**2, c15[p]**2, c16[p]**2, c17[p]**2, c18[p]**2, c19[p]**2, c20[p]**2, c21[p]**2]
    df_statistics['R^2'] = a


    a = []
    c1 = stats.linregress(df['Creditability'], df['Creditability'])
    # slope, intercept, r_value, p_value, std_err = stats.linregress(df['Duration_of_Credit_month'], df['Creditability'])
    c2 = stats.linregress(df['Account Balance'], df['Creditability'])
    c3 = stats.linregress(df['Duration_of_Credit_month'], df['Creditability'])
    c4 = stats.linregress(df['Payment Status of Prev. C.'], df['Creditability'])
    c5 = stats.linregress(df['Purpose'], df['Creditability'])
    c6 = stats.linregress(df['Credit_Amount'], df['Creditability'])
    c7 = stats.linregress(df['Value Savings/Stocks'], df['Creditability'])
    c8 = stats.linregress(df['Length of current employment'], df['Creditability'])
    c9 = stats.linregress(df['Instalment per cent'], df['Creditability'])
    c10 = stats.linregress(df['SexandMarital_Status'], df['Creditability'])
    c11 = stats.linregress(df['Guarantors'], df['Creditability'])
    c12 = stats.linregress(df['Duration in Current address'], df['Creditability'])
    c13 = stats.linregress(df['Most valuable asset'], df['Creditability'])
    c14 = stats.linregress(df['Age'], df['Creditability'])
    c15 = stats.linregress(df['Concurrent Credits'], df['Creditability'])
    c16 = stats.linregress(df['Type of apartment'], df['Creditability'])
    c17 = stats.linregress(df['No of Credits at this Bank'], df['Creditability'])
    c18 = stats.linregress(df['Occupation'], df['Creditability'])
    c19 = stats.linregress(df['No of dependents'], df['Creditability'])
    c20 = stats.linregress(df['Telephone'], df['Creditability'])
    c21 = stats.linregress(df['Foreign Worker'], df['Creditability'])
    p = 3
    a = ["{:.6f}".format(float(c1[p])), c2[p], c3[p], c4[p], c5[p], c6[p], c7[p], c8[p], c9[p], c10[p], c11[p], c12[p],
         c13[p], c14[p], c15[p], c16[p], c17[p], c18[p], c19[p], c20[p], c21[p]]
    df_statistics['pval'] = a
    print(df_statistics)

    return df_statistics


def df_stats2(df_predictions):
    df_statistics_predictions = pd.DataFrame()
    df_statistics_predictions['Mean'] = df_predictions.mean()
    df_statistics_predictions['Median'] = df_predictions.median()
    df_statistics_predictions['Max'] = df_predictions.max()
    df_statistics_predictions['Min'] = df_predictions.min()
    df_statistics_predictions['Stand. dev.'] = df_predictions.std()

    return df_statistics_predictions


def training(df_gesamt, i):
    df_training = pd.DataFrame()
    df_training = df_gesamt.head(i)
    df_coeff = pd.DataFrame()
    # first xvalues include no sex, second no Age, third no foreign worker, fourth not 3, fifth is complete
    #xvalues = df_training[['Account Balance', 'Duration_of_Credit_month', 'Payment Status of Prev. C.', 'Value Savings/Stocks', 'Length of current employment', 'Instalment per cent', 'Most valuable asset', 'Age', 'Concurrent Credits', 'Foreign Worker']]
    #xvalues = df_training[['Account Balance', 'Duration_of_Credit_month', 'Payment Status of Prev. C.', 'Value Savings/Stocks', 'Length of current employment', 'Instalment per cent', 'SexandMarital_Status', 'Most valuable asset', 'Concurrent Credits', 'Foreign Worker']]
    #xvalues = df_training[['Account Balance', 'Duration_of_Credit_month', 'Payment Status of Prev. C.', 'Value Savings/Stocks', 'Length of current employment', 'Instalment per cent', 'SexandMarital_Status', 'Most valuable asset', 'Age', 'Concurrent Credits']]
    xvalues = df_training[['Account Balance', 'Duration_of_Credit_month', 'Payment Status of Prev. C.', 'Value Savings/Stocks', 'Length of current employment', 'Instalment per cent', 'Most valuable asset', 'Concurrent Credits']]
    #xvalues = df_training[['Account Balance', 'Duration_of_Credit_month', 'Payment Status of Prev. C.', 'Value Savings/Stocks', 'Length of current employment', 'Instalment per cent', 'SexandMarital_Status', 'Most valuable asset', 'Age', 'Concurrent Credits', 'Foreign Worker']]
    yvalues = df_training['Creditability']
    mult_lin_reg = linear_model.LinearRegression()
    mult_lin_reg.fit(xvalues.values, yvalues.values)
    print("model score lin: %.3f" % mult_lin_reg.score(xvalues, yvalues))

    '''lin_reg = linear_model.LinearRegression()
    lin_reg.fit(df_gesamt[['Account Balance', 'Duration_of_Credit_month', 'Payment Status of Prev. C.', 'Value Savings/Stocks', 'Length of current employment', 'Instalment per cent', 'SexandMarital_Status', 'Most valuable asset', 'Age', 'Concurrent Credits', 'Foreign Worker']], df_gesamt['Creditability'])

    df_coeff['Attributs'] = ['Account Balance', 'Duration_of_Credit_month', 'Payment Status of Prev. C.', 'Value Savings/Stocks', 'Length of current employment', 'Instalment per cent', 'SexandMarital_Status', 'Most valuable asset', 'Age', 'Concurrent Credits', 'Foreign Worker']
    df_coeff['coeff'] = lin_reg.coef_
    with open('df_coeff.textmate', 'w') as file:
        file.write(str(df_coeff) + '\n')
    #print(lin_reg.coef_)'''

    return mult_lin_reg


def training2(df_gesamt, i):
    df_training_log = pd.DataFrame()
    df_training_log = df_gesamt.head(i)
    # first xvalues include no sex, second no Age, third no foreign worker, fourth not 3, fifth is complete
    #xvalues = df_training_log[['Account Balance', 'Duration_of_Credit_month', 'Payment Status of Prev. C.', 'Value Savings/Stocks', 'Length of current employment', 'Instalment per cent', 'Most valuable asset', 'Age', 'Concurrent Credits', 'Foreign Worker']]
    #xvalues = df_training_log[['Account Balance', 'Duration_of_Credit_month', 'Payment Status of Prev. C.', 'Value Savings/Stocks', 'Length of current employment', 'Instalment per cent', 'SexandMarital_Status', 'Most valuable asset', 'Concurrent Credits', 'Foreign Worker']]
    #xvalues = df_training_log[['Account Balance', 'Duration_of_Credit_month', 'Payment Status of Prev. C.', 'Value Savings/Stocks', 'Length of current employment', 'Instalment per cent', 'SexandMarital_Status', 'Most valuable asset', 'Age', 'Concurrent Credits']]
    xvalues = df_training_log[['Account Balance', 'Duration_of_Credit_month', 'Payment Status of Prev. C.', 'Value Savings/Stocks', 'Length of current employment', 'Instalment per cent', 'Most valuable asset', 'Concurrent Credits']]
    #xvalues = df_training_log[['Account Balance', 'Duration_of_Credit_month', 'Payment Status of Prev. C.', 'Value Savings/Stocks', 'Length of current employment', 'Instalment per cent', 'SexandMarital_Status', 'Most valuable asset', 'Age', 'Concurrent Credits', 'Foreign Worker']]
    yvalues = df_training_log['Creditability']
    mult_log_reg = LogisticRegression()
    mult_log_reg.fit(xvalues.values, yvalues.values)
    print("model score log: %.3f" % mult_log_reg.score(xvalues, yvalues))
    return mult_log_reg


def testing(df_gesamt, i, mult_lin_reg, mult_log_reg):
    real_creditability = []
    test_results = []
    test_binary_results = []
    test_binary_results_log = []
    test_results_log = []
    test_binary_results_log2 = []

    for j in range(i + 1, 1000):
        real_creditability.append(df_gesamt.Creditability[j])
    del df_gesamt['Creditability']

    for k in range(i + 1, 1000):
        values = df_gesamt.values[k]
        prediction = mult_lin_reg.predict([values])

        if float(prediction) > 0.5:
            test_results.append(min(float(prediction), 1))
        else:
            test_results.append(max(float(prediction), 0))
        if float(prediction) > 0.5:
            test_binary_results.append(int(round(min(float(prediction), 1), 0)))
        else:
            test_binary_results.append(int(round(max(float(prediction), 0), 0)))

    for l in range(i + 1, 1000):
        values = df_gesamt.values[l]
        prediction_log = mult_log_reg.predict([values])
        test_binary_results_log.append(int(prediction_log))

    for m in test_results:
        p = math.exp(m)/(math.exp(m)+1)
        test_results_log.append(p)
        #print(p)

        if float(p) > 0.5:
            test_binary_results_log2.append(int(round(min(float(p), 1), 0)))
        else:
            test_binary_results_log2.append(int(round(max(float(p), 0), 0)))


    df_testresults = pd.DataFrame()
    df_testresults['real_creditability'] = real_creditability
    df_testresults['test_results_lin'] = test_results
    #df_testresults['test_results_log'] = test_results_log
    df_testresults['test_binary_results_lin'] = test_binary_results
    df_testresults['test_binary_results_log'] = test_binary_results_log
    #df_testresults['test_binary_results_log2'] = test_binary_results_log2

    return df_testresults


def correlation(df_gesamt):
    corr_mlr = df_gesamt.corr(method='pearson')
    plt.figure(figsize=(20, 6))
    sns.heatmap(corr_mlr, annot=True, cmap='Blues')
    plt.show()

    return corr_mlr


def rmse_lin(df_testresults, rmse_res):
    predictedVals = df_testresults.test_results_lin
    realVals = df_testresults.real_creditability
    rrmse = mean_squared_error(realVals, predictedVals)**0.5
    rmse_res.append(rrmse)

    return rrmse


'''def rmse_log(df_testresults, rmse_res1):
    predictedVals = df_testresults.test_results_log
    realVals = df_testresults.real_creditability
    rrmse = mean_squared_error(realVals, predictedVals)**0.5
    rmse_res1.append(rrmse)

    return rrmse'''


def rmse_bin_lin(df_testresults, rmse_res2):
    predictedVals = df_testresults.test_binary_results_lin
    realVals = df_testresults.real_creditability
    rrmse = mean_squared_error(realVals, predictedVals)**0.5
    rmse_res2.append(rrmse)

    return rrmse


def rmse_bin_log(df_testresults, rmse_res3):
    predictedVals = df_testresults.test_binary_results_log
    realVals = df_testresults.real_creditability
    rrmse = mean_squared_error(realVals, predictedVals)**0.5
    rmse_res3.append(rrmse)

    return rrmse


def plottingrmse(training_rows, rmse_result, rmse_result1, rmse_result2, rmse_result3):
    plt.plot(training_rows, rmse_result, label='lin. regression (decimal results)', color='darkblue')
    #plt.plot(training_rows, rmse_result1, label='log. regression (decimal results)', color='red')
    plt.plot(training_rows, rmse_result2, label='lin. regression (binary results)', color='orange')
    plt.plot(training_rows, rmse_result3, label='log. regression (binary results)', color='skyblue')
    plt.legend()
    plt.title('Dependence of RMSE with size of training dataset')
    plt.xlabel('Size of training dataset')
    plt.ylabel('RMSE')
    plt.show()


def confusionmatrix(df_results):
    y_test = df_results['real_creditability']
    y_pred = df_results['test_binary_results_lin']
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(cnf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion matrix: linear regression (binary results)')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()


def confusionmatrix2(df_results):
    y_test = df_results['real_creditability']
    y_pred = df_results['test_binary_results_log']
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(cnf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion matrix: logistic regression (binary results)')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()


def correlationmatrix(df_results):
    corr_mlr = df_results.corr(method='pearson')
    plt.figure(figsize=(20, 6))
    sns.heatmap(corr_mlr, annot=True, cmap='Blues')
    plt.title('Correlation matrix')
    plt.show()


def roccurve(df_results):
    y_test = df_results['real_creditability']
    y_pred = df_results['test_results_lin']
    abc, abcd, _ = metrics.roc_curve(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    #y_test = df_results['real_creditability']
    #y_pred0 = df_results['test_results_log']
    #abc0, abcd0, _ = metrics.roc_curve(y_test, y_pred0)
    #auc0 = metrics.roc_auc_score(y_test, y_pred0)
    y_test = df_results['real_creditability']
    y_pred1 = df_results['test_binary_results_lin']
    abc1, abcd1, _ = metrics.roc_curve(y_test, y_pred1)
    auc1 = metrics.roc_auc_score(y_test, y_pred1)
    y_test = df_results['real_creditability']
    y_pred2 = df_results['test_binary_results_log']
    abc2, abcd2, _ = metrics.roc_curve(y_test, y_pred2)
    auc2 = metrics.roc_auc_score(y_test, y_pred2)
    plt.plot(abc, abcd, label="roc_curve_lin=" + str(auc), color='darkblue')
    #plt.plot(abc0, abcd0, label="roc_curve_log=" + str(auc0), color='red')
    plt.plot(abc1, abcd1, label="roc_curve_lin_binary=" + str(auc1), color='orange')
    plt.plot(abc2, abcd2, label="roc_curve_log_binary=" + str(auc2), color='skyblue')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=4)
    plt.show()


def acc(df_results):
    y_test = df_results['real_creditability']
    y_pred = df_results['test_binary_results_lin']
    acc = metrics.accuracy_score(y_test, y_pred)

    return acc


def prec(df_results):
    prec = []
    y_test = df_results['real_creditability']
    y_pred = df_results['test_binary_results_lin']
    prec = metrics.precision_score(y_test, y_pred)

    return prec


def rec(df_results):
    rec = []
    y_test = df_results['real_creditability']
    y_pred = df_results['test_binary_results_lin']
    rec = metrics.recall_score(y_test, y_pred)

    return rec


def acc2(df_results):
    y_test = df_results['real_creditability']
    y_pred = df_results['test_binary_results_log']
    acc = metrics.accuracy_score(y_test, y_pred)

    return acc


def prec2(df_results):
    prec = []
    y_test = df_results['real_creditability']
    y_pred = df_results['test_binary_results_log']
    prec = metrics.precision_score(y_test, y_pred)

    return prec


def rec2(df_results):
    rec = []
    y_test = df_results['real_creditability']
    y_pred = df_results['test_binary_results_log']
    rec = metrics.recall_score(y_test, y_pred)

    return rec


main()
