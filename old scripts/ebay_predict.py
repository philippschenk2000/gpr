# formalization & visualization
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
# Common Model Helpers
from sklearn import metrics
from sklearn import model_selection
# Common Model Algorithms
from sklearn import tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis
from sklearn.feature_selection import mutual_info_regression
# Common Model Evaluations
from sklearn.metrics import confusion_matrix, roc_curve  # To evaluate our model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score  # to split the data
from xgboost import XGBClassifier
plt.rcParams["font.family"] = "Times New Roman"



def main():
    warnings.filterwarnings("ignore")
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    data = pd.read_csv('csv_data/final_data.csv')
    data = data.sample(frac=1).reset_index(drop=True)
    pic_count = []
    for l in data['pics']:
        l = l.replace('[', '')
        l = l.replace(']', '')
        l = l.replace("'", '')
        l = l.replace('"', '')
        l = l.split(',')
        pic_count.append(len(l))
    data['pic_count'] = pic_count
    #print(data['pic_count'].head())

    # 1 CORRECTING DATA
    for mmm in range(0, 1):
        del data['pics'], data['titles'], data['dates'], data['descriptions'], data['fraud_x'], data['scraptime'], data['STADT_x'], data['BUNDESLAND_x'], data['product'], data['product_model'], data['text_x'], data['STADT_y'], data['BUNDESLAND_y'], data['NUTS3'], data['NUTS2'], data['text_y'], data['fraud_x.1'], data['fraud_y.1'], data['fraud_y'], data['prices_y'], data['west'], data['qkm'], data['capacity'], data['einwohner'], data['offer_day']
        data.rename(columns={'numb_share_x': 'title_numb_share', 'exclam_mark_share_x': 'title_exclam_mark_share', 'upper_share_x': 'title_upper_share', 'emojis_share_x': 'title_emojis_share', 'emojis_x': 'title_emojis', 'prices_x': 'prices'}, inplace=True)
        data.rename(columns={'length': 'title_length', 'numb_count': 'title_numb_count', 'exclam_mark_count': 'title_exclam_mark_count', 'upper_count': 'title_upper_count'}, inplace=True)
        data.rename(columns={'numb_share_y': 'desc_numb_share', 'exclam_mark_share_y': 'desc_exclam_mark_share', 'upper_share_y': 'desc_upper_share', 'emojis_share_y': 'desc_emojis_share', 'emojis_y': 'desc_emojis'}, inplace=True)
        data.rename(columns={'characters': 'desc_characters', 'numbs': 'desc_numbs', 'exclam_marks': 'desc_exclam_marks', 'uppers': 'desc_uppers', 'words': 'desc_words', 'spacy_spelling': 'desc_spacy_spelling', 'paypal': 'desc_paypal', 'paypal_freunde': 'desc_paypal_freunde', 'mistake_rate': 'desc_mistake_rate'}, inplace=True)
        #data = data.loc[(data['prices'] > 50) & (data['prices'] < 2000) & (data['PLZ'] < 99999)]
        cols = list(data)
        cols[-1], cols[0] = cols[0], cols[-1]
        data = data.reindex(columns=cols)
        del data['desc_words'], data['PLZ']
        print(round(data['fraud'].value_counts()[1] / len(data) * 100, 2), '% Frauds in the complete dataset')
    #data = data.head(1000)


    # 2 SPLITTING THE DATA & INPUTS
    X = data.drop('fraud', axis=1)
    y = data['fraud']
    test_size = 0.3
    threshold = 0.29
    profit_max_mla = 'ExtraTreesClassifier' #ExtraTreesClassifier 'XGBClassifier'
    profit_pred0_act0 = 5
    profit_pred1_act0 = -25-5
    profit_pred0_act1 = -50
    profit_pred1_act1 = 50
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    for mmm in range(0, 1):
        print('-' * 10)
        print(len(X_train), 'offers in trainset; ', len(X_test), 'offers in testset')
        print(round(y_train.value_counts()[1] / len(y_train) * 100, 2), '% Frauds in the trainset')
        print(round(y_test.value_counts()[1] / len(y_test) * 100, 2), '% Frauds in the testset')
        offer_ID = X_test['offer-ID'].reset_index(drop=True)
        X_test = X_test.drop('offer-ID', axis=1)
        X_train = X_train.drop('offer-ID', axis=1)


    # 3 STATS
    for x in range(0, 1):
        break
        fig, ax = plt.subplots(figsize=(15, 12))
        del data['offer-ID']
        cols = list(data)
        cols[-1], cols[0] = cols[0], cols[-1]
        data = data.reindex(columns=cols)
        corr = data.corr()
        custom_palette = sns.color_palette("coolwarm_r", 24, as_cmap=True)
        sns.heatmap(corr, cmap=custom_palette, annot_kws={'size': 20}, vmin=-0.3, vmax=0.4)
        plt.title("Correlation Matrix \n (use for reference)", fontsize=14)
        plt.show()
    df_statistics = pd.DataFrame()
    for mm in range(0, 1):
        df_statistics['mean'] = np.around(X.mean(), 3)
        df_statistics['mean'] = np.around(df_statistics['mean'], 3)
        df_statistics['max'] = X.max()
        df_statistics['min'] = X.min()
        df_statistics['std'] = np.around(X.std(), 3)
        df_statistics['std'] = np.around(df_statistics['std'], 3)

        # slope, intercept, r_value, p_value, std_err = stats.linregress(df['columnxy'], df['fraud'])
        r_value = []
        p_value = []
        for i in X.columns:
            if X[str(i)].dtypes == int or X[str(i)].dtypes == float:
                c = stats.linregress(X[str(i)], y)
                r_value.append("{:.4f}".format(float(c[2])))
                p_value.append("{:.4f}".format(float(c[3])))
                #print(chisquare(np.array(X[str(i)], y).T))
        df_statistics['r_value'] = r_value
        df_statistics['p_value'] = p_value
        df_statistics['r_value'] = df_statistics['r_value'].astype(float)
        importances = mutual_info_regression(X, y)
        feat_importances = pd.Series(importances, X.columns[0:len(X.columns)])
        df_statistics['information_gain'] = feat_importances
        df_statistics = df_statistics.sort_values(by='information_gain', ascending=True)
        #df_statistics['information_gain'].plot(kind='barh')


    # 4 MACHINE LEARNING ALGORITHMS
    MLA = [     # Ensemble Methods
                #ensemble.AdaBoostClassifier(),
                #ensemble.BaggingClassifier(),
                ensemble.ExtraTreesClassifier(),
                #ensemble.GradientBoostingClassifier(),
                ensemble.RandomForestClassifier(),
                # GLM
                linear_model.LogisticRegressionCV(),
                # Navies Bayes
                naive_bayes.BernoulliNB(),
                # Nearest Neighbor
                neighbors.KNeighborsClassifier(),
                # Trees
                #tree.DecisionTreeClassifier(),
                #tree.ExtraTreeClassifier(),
                # Discriminant Analysis
                discriminant_analysis.LinearDiscriminantAnalysis(),
                # xgboost
                XGBClassifier()]


    # 5 CROSS VALIDATION
    for x in range(0, 1):
        print('-' * 10)
        break
        # evaluate each model in turn
        results = []
        names = []
        for model in MLA:
            kfold = KFold(n_splits=12)
            cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc')
            results.append(cv_results)
            model_short = str(model)[:13]
            print('Mean ROC AUC: %.3f' % cv_results.mean() + ', ' + str(model_short))
            names.append(model_short)
        # boxplot algorithm comparison
        fig = plt.figure(figsize=(11, 6))
        fig.suptitle('Algorithm Comparison: ' + 'roc_auc')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.show()
    '''
    # CROSS VALIDATION - BALANCED DATASET - OVERSAMPLING
    results2 = []
    for model in MLA:
        break # makes no difference to non oversampled
        k_values = [1, 2, 3, 4, 5, 6, 7]
        for k in k_values:
            results = []
            over = SMOTE(sampling_strategy=0.1, k_neighbors=k)
            under = RandomUnderSampler(sampling_strategy=0.5)
            #model = ensemble.ExtraTreesClassifier()
            steps = [('over', over), ('under', under), ('model', model)]
            pipeline = Pipeline(steps=steps)
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            scores = cross_val_score(pipeline, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
            results.append(scores.mean())
            results2.append(sum(results)/len(results))
            print('Mean ROC AUC after oversampling: %.3f' % scores.mean() + ', ' + str(model))
    '''


    # 6 FIND THE BEST PARAMETERS FOR PREDICTORS
    cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6, random_state=0)
    for model in MLA:
        break
        print(model)
        base_results = model_selection.cross_validate(model, X_train, y_train, cv=cv_split, scoring='roc_auc')
        #print('Before GridSearchCV Training Shape Old: ', X_train.shape)
        print("Before GridSearchCV Test w/bin score mean: {:.2f}".format(base_results['test_score'].mean() * 100))
        print("Before GridSearchCV Test w/bin score 3*std: +/- {:.2f}".format(base_results['test_score'].std() * 100 * 3))

        grid_n_estimator = [10, 50, 100, 300]
        grid_ratio = [.1, .25, .5, .75, 1.0]
        grid_learn = [.01, .03, .05, .1, .25]
        grid_max_depth = [2, 4, 6, 8, 10, None]
        grid_min_samples = [5, 10, .03, .05, .10]
        grid_criterion = ['gini', 'entropy']
        grid_bool = [True, False]
        grid_seed = [0]
        grid_solver = ['svd', 'lsqr', 'eigen']
        if 'AdaBoost' in str(model):
            submit_abc = model_selection.GridSearchCV(model, param_grid={'n_estimators': grid_n_estimator, 'learning_rate': grid_ratio, 'algorithm': ['SAMME', 'SAMME.R'], 'random_state': grid_seed}, scoring='roc_auc', cv=cv_split)
            submit_abc.fit(X_train, y_train)
            print('After GridSearchCV Best Parameters: ', submit_abc.best_params_) #Best Parameters:
            print('After GridSearchCV Best Score: ', submit_abc.best_score_)
        elif 'Bagging' in str(model):
            submit_bc = model_selection.GridSearchCV(model, param_grid={'n_estimators': grid_n_estimator, 'max_samples': grid_ratio, 'oob_score': grid_bool, 'random_state': grid_seed}, scoring='roc_auc', cv=cv_split)
            submit_bc.fit(X_train, y_train)
            print('After GridSearchCV Best Parameters: ', submit_bc.best_params_)
            print('After GridSearchCV Best Score: ', submit_bc.best_score_)
        elif 'ExtraTrees' in str(model):
            submit_etc = model_selection.GridSearchCV(model, param_grid={'n_estimators': grid_n_estimator, 'criterion': grid_criterion, 'max_depth': grid_max_depth, 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)
            submit_etc.fit(X_train, y_train)
            print('After GridSearchCV Best Parameters: ', submit_etc.best_params_)
            print('After GridSearchCV Best Score: ', submit_etc.best_score_)
        elif 'GradientBoost' in str(model):
            submit_gbc = model_selection.GridSearchCV(model, param_grid={'learning_rate': grid_ratio, 'n_estimators': grid_n_estimator, 'max_depth': grid_max_depth, 'random_state': grid_seed}, scoring='roc_auc', cv=cv_split)
            submit_gbc.fit(X_train, y_train)
            print('After GridSearchCV Best Parameters: ', submit_gbc.best_params_)
            print('After GridSearchCV Best Score: ', submit_gbc.best_score_)
        elif 'RandomForest' in str(model):
            submit_rfc = model_selection.GridSearchCV(model, param_grid={'n_estimators': grid_n_estimator, 'criterion': grid_criterion, 'max_depth': grid_max_depth, 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)
            submit_rfc.fit(X_train, y_train)
            print('After GridSearchCV Best Parameters: ', submit_rfc.best_params_)
            print('After GridSearchCV Best Score: ', submit_rfc.best_score_)
        elif 'GaussianProcessClassifier' in str(model):
            submit_gpc = model_selection.GridSearchCV(model, param_grid={'max_iter_predict': grid_n_estimator, 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)
            submit_gpc.fit(X_train, y_train)
            print('After GridSearchCV Best Parameters: ', submit_gpc.best_params_)
            print('After GridSearchCV Best Score: ', submit_gpc.best_score_)
        elif 'LogisticRegressionCV' in str(model):
            submit_lrc = model_selection.GridSearchCV(model, param_grid={'fit_intercept': grid_bool, 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)
            submit_lrc.fit(X_train, y_train)
            print('After GridSearchCV Best Parameters: ', submit_lrc.best_params_)
            print('After GridSearchCV Best Score: ', submit_lrc.best_score_)
        elif 'BernoulliNB' in str(model):
            submit_bnbc = model_selection.GridSearchCV(model, param_grid={'alpha': grid_ratio}, scoring = 'roc_auc', cv = cv_split)
            submit_bnbc.fit(X_train, y_train)
            print('After GridSearchCV Best Parameters: ', submit_bnbc.best_params_)
            print('After GridSearchCV Best Score: ', submit_bnbc.best_score_)
        elif 'KNeighborsClassifier' in str(model):
            submit_knnc = model_selection.GridSearchCV(model, param_grid={'n_neighbors': [1,2,3,4,5,6,7], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}, scoring = 'roc_auc', cv = cv_split)
            submit_knnc.fit(X_train, y_train)
            print('After GridSearchCV Best Parameters: ', submit_knnc.best_params_)
            print('After GridSearchCV Best Score: ', submit_knnc.best_score_)
        elif 'LinearDiscriminantAnalysis' in str(model):
            submit_lda = model_selection.GridSearchCV(model, param_grid={'solver': grid_solver}, scoring = 'roc_auc', cv = cv_split)
            submit_lda.fit(X_train, y_train)
            print('After GridSearchCV Best Parameters: ', submit_lda.best_params_)
            print('After GridSearchCV Best Score: ', submit_lda.best_score_)
        elif 'XGBClassifier' in str(model):
            submit_xgb = model_selection.GridSearchCV(model, param_grid= {'learning_rate': grid_learn, 'max_depth': [0,2,4,6,8,10], 'n_estimators': grid_n_estimator, 'seed': grid_seed}, scoring = 'roc_auc', cv = cv_split)
            submit_xgb.fit(X_train, y_train)
            print('After GridSearchCV Best Parameters: ', submit_xgb.best_params_)
            print('After GridSearchCV Best Score: ', submit_xgb.best_score_)
        print('-' * 10)


    # 7 MACHINE LEARNING ALGORITHMS WITH OPTIMAL PARAMS
    MLA = [
           #ensemble.AdaBoostClassifier(algorithm='SAMME', learning_rate=0.25, n_estimators=300, random_state=0),  #{'algorithm': 'SAMME', 'learning_rate': 0.25, 'n_estimators': 300, 'random_state': 0}
           #ensemble.BaggingClassifier(max_samples=0.5, n_estimators=300, oob_score=True, random_state=0),  #{'max_samples': 0.5, 'n_estimators': 300, 'oob_score': True, 'random_state': 0}
           ensemble.ExtraTreesClassifier(criterion='entropy', max_depth=10, n_estimators=300, random_state=0),  #{'criterion': 'entropy', 'max_depth': 10, 'n_estimators': 300, 'random_state': 0}
           #ensemble.GradientBoostingClassifier(learning_rate=0.1, n_estimators=100,min_samples_split=500,random_state=0),  #{'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 100, 'random_state': 0}
           ensemble.RandomForestClassifier(criterion='entropy', max_depth=8, n_estimators=300, random_state=0),  #{'criterion': 'entropy', 'max_depth': 8, 'n_estimators': 300, 'random_state': 0}
           linear_model.LogisticRegressionCV(fit_intercept=True, random_state=0, solver='newton-cg'),  # {'fit_intercept': True, 'random_state': 0, 'solver': 'newton-cg'}
           naive_bayes.BernoulliNB(alpha=1.0),  #{'alpha': 1.0}
           neighbors.KNeighborsClassifier(algorithm='ball_tree', n_neighbors=7, weights='distance'),  # {'algorithm': 'ball_tree', 'n_neighbors': 7, 'weights': 'distance'}
           discriminant_analysis.LinearDiscriminantAnalysis(solver='svd'), #{'solver': 'svd'}
           XGBClassifier(learning_rate=0.05, max_depth=2, n_estimators=300, seed=0) #{'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'seed': 0}
            ]


    # 8 ANALYSIS - PREDICTING
    for u in range(0, 1):
        print('-' * 10)
        df = pd.DataFrame()
        df_roc = pd.DataFrame()
        df_results = pd.DataFrame()
        X_test = X_test.reset_index()
        y_test = y_test.reset_index()
        del X_test['index'], y_test['index']
        for j in range(0, len(MLA)):
            mla_now = MLA[j]
            #if 'Bagging' not in str(mla_now):
                #xy = feature_selection.RFECV(mla_now, step=1, scoring='roc_auc', cv=cv_split)
                #xy.fit(X_train, y_train)
                #X_rfe = X_train.columns.values[xy.get_support()]
                #print('AFTER DT RFE Training Columns New: ', X_rfe)
            mla_now.fit(X_train.values, y_train.values)
            #y_pred_train = mla_now.predict(X_train)
            y_pred_prob = mla_now.predict_proba(X_test)[:, 1]
            if 'RandomForestClassifier' in str(mla_now) or 'ExtraTreesClassifier' in str(mla_now) or 'XGBClassifier' in str(mla_now):
                feature = np.around(mla_now.feature_importances_, 4)
                feature = pd.Series(feature, X_train.columns)
                df_statistics[str(mla_now)[:30]] = feature
                df_statistics = df_statistics.sort_values(by=str(mla_now)[:30], ascending=False)
            if 'RandomForestClassifier' in str(mla_now):
                fn = X_train.columns
                #fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10, 13), dpi=300)
                #tree.plot_tree(mla_now.estimators_[0], feature_names=fn, filled=True)
                #fig.savefig('rf_individualtree.png')

            df['y_pred_prob'] = y_pred_prob
            res = []
            for l in df['y_pred_prob']:
                if l > threshold:
                    res.append(1)
                else:
                    res.append(0)
            df['y_predicted'] = res
            df['y_test'] = y_test['fraud']
            df['offer-ID'] = offer_ID
            df['MLA'] = str(mla_now)[:20]
            #df['accuracy_train_set'] = round(float(metrics.accuracy_score(y_train, y_pred_train)*100), 2)
            #df['accuracy_test_set'] = round(float(metrics.accuracy_score(y_test, res)*100), 2)
            df['rmse'] = round(mean_squared_error(y_test, y_pred_prob)**0.5, 3)
            df_results = df_results.append(df, ignore_index=True)
            df_roc1 = pd.DataFrame()

            fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
            auc = metrics.roc_auc_score(y_test, y_pred_prob)
            df_roc1['fpr'] = fpr
            df_roc1['tpr'] = tpr
            df_roc1['thresholds'] = thresholds
            df_roc1['auc'] = auc
            df['auc'] = round((auc*100), 2)
            print(df.head(1))
            df_roc1['mla_now'] = str(mla_now)
            df_roc = df_roc.append(df_roc1, ignore_index=True)
        profit = profit_calc(df_results, profit_pred0_act0, profit_pred1_act0, profit_pred0_act1, profit_pred1_act1)
        roccurve = rocking_curve(df_roc)
        confuse = confusionmatrix(df_results, profit_max_mla)
        with open('stats_final_data.textmate', 'w') as file:
            file.write(str(df_statistics) + '\n')
        df_results_short = df_results.loc[df_results['offer-ID'] == offer_ID[int(len(data) * 1 * test_size - 2)]].reset_index(drop=True)
        df_results_short = df_results_short.sort_values(by='auc', ascending=False)
        with open('results.textmate', 'w') as file:
            file.write(str(df_results_short) + '\n')
        with open('all_results.textmate', 'w') as file:
            file.write(str(df_results) + '\n')



        # PLOTTING
        if test_size < 0.5:
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.barplot(data=df_results_short, x="rmse", y="MLA", color='orange')
            plt.title("rmse by MLA")
            #plt.show()



def confusionmatrix(df_results, profit_max_mla):
    df_results['MLA'] = df_results['MLA'].astype(str)
    mla = []
    for i in df_results['MLA']:
        if 'AdaBoostCla' in i:
            i = 'AdaBoostClassifier'
        elif 'BaggingCla' in i:
            i = 'BaggingClassifier'
        elif 'ExtraTreesCla' in i:
            i = 'ExtraTreesClassifier'
        elif 'GradientBoost' in i:
            i = 'GradientBoostingClassifier'
        elif 'RandomForestC' in i:
            i = 'RandomForestClassifier'
        elif 'LogisticRegr' in i:
            i = 'LogisticRegressionCV'
        elif 'BernoulliNB' in i:
            i = 'BernoulliNB'
        elif 'KNeighborsCl' in i:
            i = 'KNeighborsClassifier'
        elif 'LinearDiscrimina' in i:
            i = 'LinearDiscriminantAnalysis'
        elif 'XGBClassifier' in i:
            i = 'XGBClassifier'
        mla.append(i)
    df_results['MLA'] = mla
    df1 = df_results.loc[df_results['MLA'] == str(profit_max_mla)]

    cnf_matrix = metrics.confusion_matrix(df1['y_test'], df1['y_predicted'])
    sns.heatmap(cnf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion matrix: ' + str(df1['MLA'][0]))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()


def rocking_curve(df_roc):
    df_roc['mla_now'] = df_roc['mla_now'].astype(str)
    mla = []
    for i in df_roc['mla_now']:
        if 'AdaBoostClassifier' in i:
            i = 'AdaBoostClassifier'
        elif 'BaggingClassifier' in i:
            i = 'BaggingClassifier'
        elif 'ExtraTreesClassifier' in i:
            i = 'ExtraTreesClassifier'
        elif 'GradientBoostingClassifier' in i:
            i = 'GradientBoostingClassifier'
        elif 'RandomForestClassifier' in i:
            i = 'RandomForestClassifier'
        elif 'LogisticRegressionCV' in i:
            i = 'LogisticRegressionCV'
        elif 'BernoulliNB' in i:
            i = 'BernoulliNB'
        elif 'KNeighborsClassifier' in i:
            i = 'KNeighborsClassifier'
        elif 'LinearDiscriminantAnalysis' in i:
            i = 'LinearDiscriminantAnalysis'
        elif 'XGBClassifier' in i:
            i = 'XGBClassifier'
        mla.append(i)
    df_roc['mla_now'] = mla

    mla_now = "AdaBoostClassifier"
    fpr1 = df_roc.loc[df_roc['mla_now'] == str(mla_now)].reset_index(drop=True)['fpr']
    tpr1 = df_roc.loc[df_roc['mla_now'] == str(mla_now)].reset_index(drop=True)['tpr']
    auc1 = round(df_roc.loc[df_roc['mla_now'] == str(mla_now)].reset_index(drop=True)['auc'], 3)
    plt.plot([0, 1], [0, 1], 'k--')
    #plt.plot(fpr1, tpr1, label=str(mla_now) + ': ' + str(auc1[0]), color='orange')
    mla_now = "ExtraTreesClassifier"
    fpr2 = df_roc.loc[df_roc['mla_now'] == str(mla_now)].reset_index(drop=True)['fpr']
    tpr2 = df_roc.loc[df_roc['mla_now'] == str(mla_now)].reset_index(drop=True)['tpr']
    auc2 = round(df_roc.loc[df_roc['mla_now'] == str(mla_now)].reset_index(drop=True)['auc'], 3)
    plt.plot(fpr2, tpr2, label=str(auc2[0]) + ', ' + str(mla_now), color='skyblue')
    mla_now = "RandomForestClassifier"
    fpr3 = df_roc.loc[df_roc['mla_now'] == str(mla_now)].reset_index(drop=True)['fpr']
    tpr3 = df_roc.loc[df_roc['mla_now'] == str(mla_now)].reset_index(drop=True)['tpr']
    auc3 = round(df_roc.loc[df_roc['mla_now'] == str(mla_now)].reset_index(drop=True)['auc'], 3)
    plt.plot(fpr3, tpr3, label=str(auc3[0]) + ', ' + str(mla_now), color='yellow')
    mla_now = "LogisticRegressionCV"
    fpr4 = df_roc.loc[df_roc['mla_now'] == str(mla_now)].reset_index(drop=True)['fpr']
    tpr4 = df_roc.loc[df_roc['mla_now'] == str(mla_now)].reset_index(drop=True)['tpr']
    auc4 = round(df_roc.loc[df_roc['mla_now'] == str(mla_now)].reset_index(drop=True)['auc'], 3)
    plt.plot(fpr4, tpr4, label=str(auc4[0]) + ', ' + str(mla_now), color='purple')
    mla_now = 'BernoulliNB'
    fpr5 = df_roc.loc[df_roc['mla_now'] == str(mla_now)].reset_index(drop=True)['fpr']
    tpr5 = df_roc.loc[df_roc['mla_now'] == str(mla_now)].reset_index(drop=True)['tpr']
    auc5 = round(df_roc.loc[df_roc['mla_now'] == str(mla_now)].reset_index(drop=True)['auc'], 3)
    plt.plot(fpr5, tpr5, label=str(auc5[0]) + ', ' + str(mla_now), color='black')
    mla_now = "KNeighborsClassifier"
    fpr6 = df_roc.loc[df_roc['mla_now'] == str(mla_now)].reset_index(drop=True)['fpr']
    tpr6 = df_roc.loc[df_roc['mla_now'] == str(mla_now)].reset_index(drop=True)['tpr']
    auc6 = round(df_roc.loc[df_roc['mla_now'] == str(mla_now)].reset_index(drop=True)['auc'], 3)
    plt.plot(fpr6, tpr6, label=str(auc6[0]) + ', ' + str(mla_now), color='grey')
    mla_now = "LinearDiscriminantAnalysis"
    fpr7 = df_roc.loc[df_roc['mla_now'] == str(mla_now)].reset_index(drop=True)['fpr']
    tpr7 = df_roc.loc[df_roc['mla_now'] == str(mla_now)].reset_index(drop=True)['tpr']
    auc7 = round(df_roc.loc[df_roc['mla_now'] == str(mla_now)].reset_index(drop=True)['auc'], 3)
    plt.plot(fpr7, tpr7, label=str(auc7[0]) + ', ' + str(mla_now), color='green')
    mla_now = "XGBClassifier"
    fpr8 = df_roc.loc[df_roc['mla_now'] == str(mla_now)].reset_index(drop=True)['fpr']
    tpr8 = df_roc.loc[df_roc['mla_now'] == str(mla_now)].reset_index(drop=True)['tpr']
    auc8 = round(df_roc.loc[df_roc['mla_now'] == str(mla_now)].reset_index(drop=True)['auc'], 3)
    plt.plot(fpr8, tpr8, label=str(auc8[0]) + ', ' + str(mla_now), color='red')


    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc=4)
    plt.show()


def profit_calc(df_results, profit_pred0_act0, profit_pred1_act0, profit_pred0_act1, profit_pred1_act1):

    df_results['MLA'] = df_results['MLA'].astype(str)
    mla = []
    for i in df_results['MLA']:
        if 'AdaBoostClassifier' in i:
            i = 'AdaBoostClassifier'
        elif 'BaggingClassifier' in i:
            i = 'BaggingClassifier'
        elif 'ExtraTreesClassifier' in i:
            i = 'ExtraTreesClassifier'
        elif 'GradientBoostingClassifier' in i:
            i = 'GradientBoostingClassifier'
        elif 'RandomForestClassifier' in i:
            i = 'RandomForestClassifier'
        elif 'LogisticRegressionCV' in i:
            i = 'LogisticRegressionCV'
        elif 'BernoulliNB' in i:
            i = 'BernoulliNB'
        elif 'KNeighborsClassifier' in i:
            i = 'KNeighborsClassifier'
        elif 'LinearDiscriminantAnalysis' in i:
            i = 'LinearDiscriminantAnalysis'
        elif 'XGBClassifier' in i:
            i = 'XGBClassifier'
        mla.append(i)
    df_results['MLA'] = mla

    df_profit_all2 = pd.DataFrame()
    mla1 = df_results.drop_duplicates(subset='MLA', keep='first')
    for x in mla1['MLA']:
        df_profit = pd.DataFrame()
        profits = []
        thresh = []
        df = df_results.loc[df_results['MLA'] == str(x)].reset_index(drop=True)
        for thresholds in range(0, 500, 1):
            threshold = thresholds/500
            res = []
            for l in df['y_pred_prob']:
                if l > threshold:
                    res.append(1)
                else:
                    res.append(0)
            conf = confusion_matrix(df['y_test'], res)
            pred0_act0 = conf[0][0]
            pred1_act0 = conf[0][1]
            pred0_act1 = conf[1][0]
            pred1_act1 = conf[1][1]
            profit = profit_pred0_act0*pred0_act0 + profit_pred1_act0*pred1_act0 + profit_pred0_act1*pred0_act1 + profit_pred1_act1*pred1_act1
            profits.append(max(0, profit)*(14800/1000000)) #35m users online per month; 35m/2361 = 14800
            thresh.append(threshold)
        df_profit['thresholds'] = thresh
        df_profit[str(x)] = profits
        df_profit_all2['thresholds'] = thresh
        print('Profit Optimum of thresholds for Classifier ', str(x), ' is: ', df_profit.loc[df_profit[str(x)] == max(df_profit[str(x)])]['thresholds'].reset_index(drop=True)[0])
        df_profit_all2 = pd.merge(df_profit_all2, df_profit, on='thresholds', how='left')

        res = []
        for l in df['y_pred_prob']:
            if l > df_profit_all2.loc[df_profit_all2[str(x)] == max(df_profit_all2[str(x)])]['thresholds'].reset_index(drop=True)[0]:
                res.append(1)
            else:
                res.append(0)
        print('accuracy at profit maximum threshold: ', round(float(metrics.accuracy_score(df['y_test'], res) * 100), 2))
        print(confusion_matrix(df['y_test'], res))
    df_profit_all = df_profit_all2.set_index('thresholds')


    plt.figure(figsize=(14, 8))
    sns.lineplot(data=df_profit_all)
    sns.scatterplot(x=df_profit_all.loc[df_profit_all['ExtraTreesClassifier'] == max(df_profit_all['ExtraTreesClassifier'])].index, y=df_profit_all['ExtraTreesClassifier'].loc[df_profit_all['ExtraTreesClassifier'] == max(df_profit_all['ExtraTreesClassifier'])].values, marker='o', s=50)
    sns.scatterplot(x=df_profit_all.loc[df_profit_all['RandomForestClassifi'] == max(df_profit_all['RandomForestClassifi'])].index, y=df_profit_all['RandomForestClassifi'].loc[df_profit_all['RandomForestClassifi'] == max(df_profit_all['RandomForestClassifi'])].values, marker='o', s=50)
    sns.scatterplot(x=df_profit_all.loc[df_profit_all['LogisticRegressionCV'] == max(df_profit_all['LogisticRegressionCV'])].index, y=df_profit_all['LogisticRegressionCV'].loc[df_profit_all['LogisticRegressionCV'] == max(df_profit_all['LogisticRegressionCV'])].values, marker='o', s=50)
    sns.scatterplot(x=df_profit_all.loc[df_profit_all['BernoulliNB'] == max(df_profit_all['BernoulliNB'])].index, y=df_profit_all['BernoulliNB'].loc[df_profit_all['BernoulliNB'] == max(df_profit_all['BernoulliNB'])].values, marker='o', s=50)
    sns.scatterplot(x=df_profit_all.loc[df_profit_all['KNeighborsClassifier'] == max(df_profit_all['KNeighborsClassifier'])].index, y=df_profit_all['KNeighborsClassifier'].loc[df_profit_all['KNeighborsClassifier'] == max(df_profit_all['KNeighborsClassifier'])].values, marker='o', s=50)
    sns.scatterplot(x=df_profit_all.loc[df_profit_all['LinearDiscriminantAn'] == max(df_profit_all['LinearDiscriminantAn'])].index, y=df_profit_all['LinearDiscriminantAn'].loc[df_profit_all['LinearDiscriminantAn'] == max(df_profit_all['LinearDiscriminantAn'])].values, marker='o', s=50)
    sns.scatterplot(x=df_profit_all.loc[df_profit_all['XGBClassifier'] == max(df_profit_all['XGBClassifier'])].index, y=df_profit_all['XGBClassifier'].loc[df_profit_all['XGBClassifier'] == max(df_profit_all['XGBClassifier'])].values, marker='o', s=50)
    sns.scatterplot(x=1, y=df_profit_all['ExtraTreesClassifier'].loc[df_profit_all['ExtraTreesClassifier'].index == max(df_profit_all['ExtraTreesClassifier'].index)], marker='o', s=200)
    sns.scatterplot(x=0, y=df_profit_all['ExtraTreesClassifier'].loc[df_profit_all['ExtraTreesClassifier'].index == min(df_profit_all['ExtraTreesClassifier'].index)], marker='o', s=200)
    plt.ylim(-5, max(df_profit_all['ExtraTreesClassifier'])*1.15)
    #plt.xlim(0, 0.7)
    plt.xlabel('Thresholds for classification')
    plt.ylabel('Profit for platform in Mio. EUR')
    plt.title('Profit Calculation by threshold (in Mio. EUR)')
    plt.legend()
    plt.show()


main()