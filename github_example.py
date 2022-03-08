# pip install git+https://github.com/MerelyMax/constrained-ga-for-fs.git#egg=constrained-ga-for-fs --upgrade

from time import time
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer

from ga_for_fs import GeneticAlgorithm

if __name__ == '__main__':
    start = time()
    features, y = load_breast_cancer(return_X_y=True)

    # Implement PCA - feature reduction method
    pca = PCA(n_components=4)
    PCA_features = pca.fit_transform(features)

    # Scale features to the range of [0,1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)

    # Augment original feature space concatenating PCA features
    constructed_f = np.concatenate((features, PCA_features), axis=1)

    # Feature selection constrained-ga-for-fs
    selector = GeneticAlgorithm(X=constructed_f,
                                y=y.astype(int),
                                estimator=RandomForestClassifier,
                                scoring='f1_macro',
                                cv=5,
                                n_population=15,
                                n_gen=20,
                                crossoverType='OnePoint',
                                mutationProb=1/constructed_f.shape[1],
                                initType='coin',
                                extraFeatures_num=4,
                                verbose=True)

    best_fitness, best_indexes, best_hyperparams = selector.startGA()

    print(
        f'best_f1_macro: {best_fitness}, best_indexes {best_indexes}, ind_best_hyperparam {best_hyperparams}')
    finish = time()
    print('')
    print(f'Passed time: {round((finish-start)/60, 4)}, min')
