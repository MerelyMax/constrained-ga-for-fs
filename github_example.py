# pip install git+https://github.com/MerelyMax/constrained-ga-for-fs.git#egg=constrained-ga-for-fs --upgrade
# from multiprocessing import set_start_method
# set_start_method("spawn")
from operator import index
from time import time

import numpy as np
from numpy.lib.function_base import average
import pandas as pd

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.decomposition import PCA, KernelPCA

from sklearn.metrics import f1_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.datasets import load_breast_cancer

from ga_for_fs import GeneticAlgorithm
# from BDE_ga import GeneticAlgorithm

# import os
# os.environ['JOBLIB_START_METHOD'] = 'forkserver'
# print(f'Curent JOBLOB environment: {os.environ.get("JOBLIB_START_METHOD")}')

def start_SVC(features_X, y, cv_folds_num=1, OOF = False):

    model = GridSearchCV(estimator=svm.SVC(max_iter=100000),
                         param_grid=[{ 'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),
                                        'C' : np.insert(np.arange(10.0, 110, 10), 0, [0.5,1,5]),
                                        'gamma' : np.arange(0.1, 1.1, 0.1)}],
                         scoring='f1_macro',
                         n_jobs=-1,
                         refit=True,
                         cv=cv_folds_num)
    model.fit(features_X, y)
    print(f'Best params found:', model.best_params_)
    # несмотря на refit=True метрика рассчитывается, как 
    # Mean cross-validated score of the best_estimator
    print(f'Best f1_macro score = {model.best_score_}')
    if (OOF == True):
        print(60*'-')
        print('Creating out-of-fold predictions...')
        print(60*'-')
        # Для stacking создадим OUT-OF-FOLD predictions
        out_of_fold_pred = np.zeros(y.shape)
        best_estimator = model.best_estimator_
        for train, test in StratifiedKFold(n_splits=cv_folds_num, shuffle=False).split(features_X, y):
            best_estimator.fit(features_X[train], y[train])
            y_pred = best_estimator.predict(features_X[test])
            out_of_fold_pred[test] = y_pred
    else:
        print(60*'-')
        print('Refitting on a whole set with best hyperparam...')
        print(60*'-')
        best_estimator = model.best_estimator_
        refitted_pred = model.predict(features_X)

    # Return best_estimator || best_score || best_params || out-of-fold y_pred // или refitted_pred
    return best_estimator, model.best_score_, model.best_params_, refitted_pred


def start_PCA(features, n_comp):
    print(60*'-')
    print('Starting PCA...')
    pca = PCA(n_components=n_comp)

    # Вот здесь отыскиваются собственные вектора/числа ковариационной матрицы
    # т.е. направления вдоль которых наибольшее количество дисперсии
    pca.fit(features)
    print(f'Explained variance ratio {pca.explained_variance_ratio_}')

    # На основании полученных направлений (нового пространства) тестируем новые данные
    PCA_features_test = pca.transform(features)
    Xhat = pca.inverse_transform(PCA_features_test)
    print(f'PCA MSE= {mean_squared_error(features, Xhat)}')
    print('')
    # После полной настройки и оценки подаем на вход все данные
    PCA_features = pca.transform(features)
    print(60*'-')
    return PCA_features


def start_kernel_PCA(features, components):
    print(60*'-')
    print('Starting kernel PCA...')
    param_grid = {'kernel': ['sigmoid', 'cosine', 'rbf'],  # poly
                  'gamma': np.arange(0.1, 3, 0.1)}

    results = []
    for kernel_ in param_grid['kernel']:
        for gamma in param_grid['gamma']:
            kpca = KernelPCA(n_components=components,
                             kernel=kernel_,
                             gamma=gamma,
                             fit_inverse_transform=True)
            X_kpca = kpca.fit_transform(features)
            X_hat = kpca.inverse_transform(X_kpca)
            results.append(
                [mean_squared_error(features, X_hat), kernel_, gamma])

    best_hyperparams = pd.DataFrame(results,
                                    columns=['MSE', 'Kernel', 'Gamma']).sort_values(by='MSE',
                                                                                    ascending=True).iloc[0]
    print('Kernel PCA')
    print(best_hyperparams)
    print('')
    kpca = KernelPCA(n_components=components,
                     kernel=best_hyperparams['Kernel'],
                     gamma=best_hyperparams['Gamma'],
                     fit_inverse_transform=True)
    X_kpca = kpca.fit_transform(features)

    # df = pd.DataFrame(np.array(X_kpca))
    # sns.pairplot(data=df)
    print(60*'-')
    return X_kpca

def start_experiment(features_X, labels, cv_folds_num, algorithm, iterations, clf_file_name):
    # Меняем тип исходных данных на numpy
    if (type(features_X) != np.ndarray) or (type(labels) != np.ndarray):
        features_X = np.array(features_X)
        labels = np.array(labels)

    results = []
    # if (best_params == None):
    best_estimator, best_score, best_params, refitted_pred = algorithm(features_X, labels, cv_folds_num)
    
    # сохраним найденный классификатор в файл
    import pickle
    with open(clf_file_name, 'wb') as f:
        pickle.dump(best_estimator, f)
    # если best_params были уже найдены (после ГА), то нужно найти
    # refited_predictions + best_score
    # else:
    #     model = estimator(C = best_params['C'],
    #                       kernel = best_params['kernel'],
    #                       gamma = best_params['gamma'],
    #                       max_iter=100000)
    #     model.fit(features_X, labels)
    #     refited_predictions = model.predict(features_X)
    #     best_score = f1_score(labels, refited_predictions, average='macro')      
    all100_OOF_predictions = []
    for i in range(iterations):
        start = time()
        print(f'Starting {i} iteration...')
        print(60*'-')
        # Оценим итоговый результат оценив точность по всей выборке
        # с использованием кросс-валидации (ощущение, что переобучение)
        cv_scores = []
        out_of_fold_pred = np.zeros(labels.shape)
        for i, (train, test) in enumerate(StratifiedKFold(n_splits=cv_folds_num, shuffle=True).split(features_X, labels)):
            best_estimator.fit(features_X[train], labels[train])
            y_pred = best_estimator.predict(features_X[test])
            out_of_fold_pred[test] = y_pred
            cv_scores.append(f1_score(labels[test], y_pred, average='macro'))
        print(f'mean f1_macro score on CV={np.mean(cv_scores)}')
        results.append((np.mean(cv_scores), best_score, best_params))
        all100_OOF_predictions.append(out_of_fold_pred)
        #------time-------
        stop = time()
        print('Execution time', stop-start, 'sec.')
        print(60*'-')
        
    final_results = pd.DataFrame(results, columns=['f1_macro (CV) - Final', 'f1_macro (CV) - Hyperparam search', 'Best params'])
    
    return final_results, all100_OOF_predictions



# точка входа в программу MAIN
if __name__ == '__main__':
    # results = []
    start = time()
    scaler = MinMaxScaler()

    features, y = load_breast_cancer(return_X_y=True)

    pca = PCA(n_components=4)
    PCA_features = pca.fit_transform(features)

    # scaler = MinMaxScaler(feature_range=(0,1))
    # features_scaled = scaler.fit_transform(features)



    constructed_f = np.concatenate((features, PCA_features), axis=1)

        # Проверь подачу y - ТИПА INT!!!
        # selector = GeneticAlgorithm(X = constructed_f,
        #                             y = german_y,
        #                             estimator = svm.SVC(),
        #                             scoring = 'f1_macro',
        #                             cv=5,
        #                             n_population=100,
        #                             n_gen=100,
        #                             crossoverType='OnePoint',
        #                             mutationProb=1/constructed_f.shape[1],
        #                             initType='uniform',
        #                             extraFeatures_num=12,
        #                             verbose=False)

        # import sys
        # sys.stdout = open('tem_results.csv', 'w')
        
    selector = GeneticAlgorithm(X = constructed_f,
                                    y = y.astype(int),
                                    estimator = RandomForestClassifier,
                                    scoring = 'f1_macro',
                                    cv=5,
                                    n_population=15,
                                    n_gen=50,
                                    crossoverType='OnePoint',
                                    mutationProb=1/constructed_f.shape[1],
                                    initType='coin',
                                    extraFeatures_num=4,
                                    verbose=False)


    best_fitness, best_indexes, best_hyperparams = selector.startGA()

    # # results.append((best_fitness, best_indexes, best_hyperparams))
        
    # pd_results = pd.DataFrame(list((best_fitness, best_indexes, best_hyperparams)), 
    #                           columns=['best_f1_macro', 'best_indexes', 'ind_best_hyperparam'])

    print('best_f1_macro: {best_fitness}, best_indexes {best_indexes}, ind_best_hyperparam {best_hyperparams}')
    finish = time()

    print('')
    print(f'Passed time: {round((finish-start)/60, 4)}, min')

    
