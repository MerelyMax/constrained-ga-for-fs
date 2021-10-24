#!/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn.model_selection import check_cv, GridSearchCV
from sklearn.metrics import check_scoring
from sklearn.model_selection._validation import _fit_and_score

#  Подумать как реализовать автоматическое определение количества экстра признаков
class GeneticAlgorithm(object):

    def __init__(self, X, y, estimator, scoring, cv, n_population, n_gen, crossoverType, mutationProb, initType, extraFeatures_num, num_features_to_init=None, indexes_prob=None, verbose=False):
        """
        Genetic Algorithm for feature selection.

        Parameters
        ----------
        X : numpy array
            Features - can be original set of features or constructed 
            (as a part of feature engineering process).

        y : numpy array
            Targets:
            -   Regression: dependent variable values, 
            -   Classification: labels.

        estimator : Scikit-learn estimator instance for regression or classification.

        scoring : string
            Any available scoring instance for scikit-learn estimator 
            (full list:https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules).

        cv : int
            Number of folds for kFold cross validation (stratified for classification).

        n_population : int
            Number of individuals in a population of genetic algorithm.

        n_gen : int
            Number of generations of genetic algorithm.

        crossoverType : string
            Crossover type of genetic algorithm. Available options: 
            -   One point: "OnePoint", 
            -   Two point: "TwoPoints",
            -   Uniform: "Uniform".

        mutationProb : float
            Mutation probability for genetic algorithm. Usual practice is to set the value to: 1/n_features.

        initType: string
            Initialization type of genetic algorithm. Available options:
            ??? "coin",
            -   "uniform", 
            -   "from_own_dist".

        indexes_prob : float, available if initType="from_own_dist", defaul=None.

        extraFeatures_num : int
            Number of additional (extra) features. Such might be extracted
            features (PCA, kernel PCA, Autoencoder, etc.) which were added to the 
            original features set. Two of four constraints of the algorithm
            will make search withing these features.

        num_features_to_init : int, default=None
            Should be used for datasets with large number of features (>30)
            to ensure convergence (the ability of the algorithm to find a solution
            under constraints).

        verbose : bool, default=False
            Regulates verbosity of the algorithm (penalty, fitness function, etc) for each epoch.
        """
        self.X = X  # Массив признаков
        self.y = y  # Массив меток классов
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.n_population = n_population  # Количество индивидов в популяции
        self.n_gen = n_gen  # Количество поколений
        self.chromosomeLength = X.shape[1]  # Длина хромосомы
        # Одноточечное, двухточечное или равномерное скрещивание
        self.crossoverType = crossoverType
        self.mutationProb = mutationProb  # Вероятность мутации
        self.initType = initType  # Тип инициализации популяции
        self.extraFeatures_num = extraFeatures_num #Additional (constructed) features number
        # Сколько в инициализации оставить признаков для очень больших выборок
        self.num_features_to_init = num_features_to_init
        # Вероятности признаков по значимости, на которые указал фильтр
        self.indexes_prob = indexes_prob
        self.verbose = verbose  # Для вывода подробной статистики

    def createPopulation(self, n_population, chromosomeLength, initType, features_to_retain=None, indexes_prob=None):
        "return matrix with population"

        if (initType == 'coin'):
            # биномиальное с возвращением (гипергеометрическое без возвращения)
            population = np.random.binomial(
                1, 0.5, size=chromosomeLength * n_population)
            population = population.reshape(n_population, chromosomeLength)

        if (initType == 'uniform'):
            population = np.zeros((n_population, chromosomeLength))
            chromosome_indexes = np.arange(chromosomeLength)

            if (features_to_retain == None):
                for k in range(n_population):
                    # сэмплирование с повторением (имитация цикла для каждого признака) || Равномерное распр.
                    samples = np.random.choice(chromosome_indexes,
                                               chromosomeLength,
                                               replace=True)
                    genes = np.unique(samples)
                    population[k] = np.isin(
                        chromosome_indexes, genes, assume_unique=True)*1
            else:
                for k in range(n_population):
                    # сэмплирование БЕЗ повторения || Равномерное распределение
                    samples = np.random.choice(chromosome_indexes,
                                               features_to_retain,
                                               replace=False)
                    genes = np.unique(samples)
                    population[k] = np.isin(
                        chromosome_indexes, genes, assume_unique=True)*1

        if (initType == 'from_own_dist'):
            population = np.zeros((n_population, chromosomeLength))
            chromosome_indexes = np.arange(chromosomeLength)

            # инициализация для маленьких выборок
            if (features_to_retain == None):
                for k in range(n_population):
                    # сэмплирование с повторением (вернем точку обратно в корзину, разыграем вероятность заново)
                    # с повторением - как бы имитация цикла по всем признакам, где на каждой итерации
                    # по известнымм вероятностям выбирается ОДИН признак. Каждый новый запуск - тоже самое
                    # что положить в корзину обратно и выбирать заново
                    samples = np.random.choice(chromosome_indexes,
                                               chromosomeLength,
                                               p=indexes_prob)
                    genes = np.unique(samples)
                    # isin создает маску - True если объект содержится в массиве
                    # *1 - умножение на 1 для конвертации True/False в 0,1
                    population[k] = np.isin(
                        chromosome_indexes, genes, assume_unique=True)*1
            else:  # инициализация для LSVT и других выборок где features>150
                two_thirds = int(features_to_retain*(2/3))
                for k in range(n_population):
                    counter = 0
                    two_thirds_indexes = []
                    # до тех пор, пока ReliefF признаков не будет 2/3
                    while counter != two_thirds:
                        # сэмплирование БЕЗ повторения
                        samples = np.random.choice(chromosome_indexes,
                                                   features_to_retain,
                                                   p=indexes_prob,
                                                   replace=False)
                        # unique_samples = np.unique(samples)
                        # считаем количество ReliefF признаков в полученном сэмпле
                        for i in range(samples.shape[0]):
                            check = any(item == samples[i]
                                        for item in indexes_prob)
                            # second condition: avoid two_thirds_indexes bigger than two_thirds
                            if (check == True) and (np.shape(two_thirds_indexes)[0] < two_thirds):
                                two_thirds_indexes.append(samples[i])
                                counter += 1

                    # setdiff1d - уберем индексы two_thirds_indexes из unique_samples
                    # разыграем эксперимент (равномерное распределение) среди оставшихся
                    # не могу сэмплировать с повторениями, тк ниже samples мб меньше, чем
                    rest_indexes = np.random.choice(np.setdiff1d(samples, two_thirds_indexes),
                                                    (features_to_retain -
                                                     two_thirds),
                                                    replace=False)
                    genes = np.concatenate((rest_indexes, two_thirds_indexes))
                    # isin создает маску - True если объект содержится в массиве
                    # *1 - умножение на 1 для конвертации True/False в 0, 1
                    population[k] = np.isin(
                        chromosome_indexes, genes, assume_unique=True)*1
        return population

    def selectionTNT(self, tntSize, n_population, fitnessValues):
        "return pair of parent indexes"
        # сгенерируем Т случайных индексов без повторений
        indexes = np.random.choice(
            range(n_population), size=tntSize, replace=False)
        # выберем в качестве первого родителя лучшего из индивидов с заданными индексами
        parent1index = indexes[np.argmax(fitnessValues[indexes])]
        # повторим для 2го родителя
        indexes = np.random.choice(
            range(n_population), size=tntSize, replace=False)
        parent2index = indexes[np.argmax(fitnessValues[indexes])]
        return parent1index, parent2index

    def crossover(self, parent1, parent2, population, chromosomeLength, crossoverType):
        if (crossoverType == 'OnePoint'):
            # выберем случайную точку разрыва
            xpoint = np.random.randint(1, chromosomeLength)
            # перемешаем гены. Т.к. возможно 2 варианата потомка, оставим только одного без сравнения их пригодности (случайно)
            offspring = np.zeros(chromosomeLength)
            offspring[0:xpoint] = population[parent1, 0:xpoint]
            offspring[xpoint:] = population[parent2, xpoint:]
        if (crossoverType == 'TwoPoints'):
            # выберем две случайных точки разрыва без повторений
            rng = np.random.default_rng()
            points = rng.choice(chromosomeLength, size=2, replace=False)
            points.sort()
            offspring = np.zeros(chromosomeLength)
            offspring[0:points[0]] = population[parent1, 0:points[0]]
            offspring[points[0]:points[1]
                      ] = population[parent2, points[0]:points[1]]
            offspring[points[1]:] = population[parent1, points[1]:]
        if (crossoverType == 'Uniform'):
            offspring = np.zeros(chromosomeLength)
            for i in range(chromosomeLength):
                # np.random.random() генерирует число от 0 до 1
                if (np.random.random() < 0.5):
                    offspring[i] = population[parent1, i]
                else:
                    offspring[i] = population[parent2, i]

        return offspring

    def mutationPoint(self, individual, mutationProba):
        n = len(individual)
        for i in range(n):
            if (np.random.random() <= mutationProba):
                individual[i] = np.abs(individual[i]-1)
        return individual

    def fitness(self, X, y, estimator, scoring, cv, individual, epoch, extraFeatures_num, verbose):
        cv = check_cv(cv, y)

        # сделаем разбиение для каждого индивида одинаковым, чтобы
        # сравнивать точность классификатора при одинаковых условиях (выборках)
        cv.random_state = 42
        cv.shuffle = False
        scorer = check_scoring(estimator, scoring=scoring)

        individual_sum = np.sum(individual, axis=0)
        if individual_sum == 0:
            scores_mean = -10000
        else:
            # Choose features according to the mask
            X_selected = X[:, np.array(individual, dtype=bool)]
            # scores = []
            # evaluation of the model
            # to guarantee there will be no overfitting - CV is used
            # for train, test in cv.split(X_selected, y):
            #     score = _fit_and_score(estimator=estimator, X=X_selected, y=y, scorer=scorer,
            #                            train=train, test=test, verbose=0, parameters=None,
            #                            fit_params=None)
            #     # simplefilter(action='ignore', category=FutureWarning)
            #     scores.append(score)
            # scores_mean = score['test_scores'] * 100
            model = GridSearchCV(estimator=estimator,
                                 param_grid=[{ 'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),
                                                'C' : np.insert(np.arange(10.0, 110, 10), 0, [0.5,1,5]),
                                                'gamma' : np.arange(0.1, 1.1, 0.1)}],
                                 scoring='f1_macro',
                                 n_jobs=-1,
                                 refit=True,
                                 cv=cv)
            model.fit(X_selected, y)
            # Mean cross-validated score of the best_estimator
            scores_mean = model.best_score_

        # Calculates extra features number in the individdual
        extraFeatures = sum(individual[len(individual)-extraFeatures_num:])
        # Calculates initial (original) features number in the individual
        originalFeatures = sum(individual[:len(individual)-extraFeatures_num])

        # наложить штраф на scores_mean по правилу:
        phi1 = max(0, (2-extraFeatures))
        phi2 = max(0, (extraFeatures-4))
        phi3 = max(0, (2-originalFeatures))
        phi4 = max(0, (originalFeatures-4))

        phi = [phi1, phi2, phi3, phi4]

        # динамический
        # penalty = dynamic_penalty(len(individual))
        # fitnessValue = scores_mean * 100 - penalty

        # Для детального анализа работы
        if (verbose == True):
            print("Epoch = ", epoch)
            print("Individual: ", individual)
            print(f"All features (sum) {individual_sum} = constructed ({extraFeatures}) + original ({originalFeatures})")
            print("phi1 = %i, phi2 = %i, phi3 = %i, phi4 = %i:" %
                  (phi1, phi2, phi3, phi4))
            print("Objective function value = ", scores_mean)
            print('')

        return scores_mean, phi

    def AdaptivePenalty(self, objective_func, violations):
        fitnessFunction = np.zeros([len(objective_func), 1])
        penalties_mas = np.zeros([len(objective_func), 1])

        avg_objective_func = objective_func.mean()
        avg_phi = [violations['phi1'].mean(), violations['phi2'].mean(),
                   violations['phi3'].mean(), violations['phi4'].mean()]
        constraints_sum = avg_phi[0] ** 2 + \
            avg_phi[1] ** 2 + avg_phi[2] ** 2 + avg_phi[3] ** 2
        for i in range(len(objective_func)):
            if ((violations.iloc[i]['phi1'] == 0) and (violations.iloc[i]['phi2'] == 0) and (violations.iloc[i]['phi3'] == 0) and (violations.iloc[i]['phi4'] == 0)):
                fitnessFunction[i] = objective_func[i]
                penalties_mas[i] = 0
            else:
                if (fitnessFunction[i] < avg_objective_func):
                    fitnessFunction[i] = avg_objective_func
                penalty = 0
                for j in range(len(avg_phi)):
                    k = (abs(avg_objective_func) * avg_phi[j]) / constraints_sum
                    penalty += k * violations.iloc[i, j]
                fitnessFunction[i] = fitnessFunction[i] - penalty
                # Сохраним штраф для вывода
                penalties_mas[i] = penalty
        return fitnessFunction, penalties_mas

    def runOneGeneration(self, population, chromosomeLength, n_population, fitnessValues, crossoverType, mutationProb):
        # newPopulation = np.empty([chromosomeLength, 1])
        newPopulation = np.zeros((n_population, chromosomeLength))
        # newPopulation = []
        for ind in range(n_population):
            # селекция
            parents = self.selectionTNT(2, n_population, fitnessValues)
            # скрещивание
            offspring = self.crossover(
                parents[0], parents[1], population, chromosomeLength, crossoverType)
            # мутация
            offspring = self.mutationPoint(offspring, mutationProb)
            # добавляем потомка в новую популяцию
            newPopulation[ind] = offspring
        return newPopulation

    def startGA(self):
        """
        Returns
        -------
        best_fitness : int
            Individual best fitness value found under constraints

        best_indexes : numpy array
            Indexes (chosen features) of the individual with 
            the best fitness value found under constraints
        """
        HallOfFame = []
        results = []
        ind_sum = []
        phi_mas = []
        statistics = []
        indexes_mask = []
        # Инициализация популяции
        population = self.createPopulation(self.n_population,
                                           self.chromosomeLength,
                                           self.initType,
                                           self.num_features_to_init)
        for currentGeneration in range(self.n_gen):
            phi = pd.DataFrame(columns=['phi1', 'phi2', 'phi3', 'phi4'])
            popObjectives = np.zeros([self.n_population, 1])
            # Расчет целевой функции каждого индивида
            for ind in range(self.n_population):
                ind_fitness = self.fitness(X=self.X,
                                           y=self.y,
                                           estimator=self.estimator,
                                           scoring=self.scoring,
                                           cv=self.cv,
                                           individual=population[ind],
                                           epoch=currentGeneration,
                                           extraFeatures_num=self.extraFeatures_num,
                                           verbose=self.verbose)

                popObjectives[ind] = ind_fitness[0]
                phi.loc[ind] = ind_fitness[1]
            # Накладываем АДАПТИВНЫЙ штраф - итоговая оценка пригодности
            popFitnesses = np.zeros([self.n_population, 1])
            penalties = np.zeros([self.n_population, 1])
            result = self.AdaptivePenalty(popObjectives, phi)
            popFitnesses = result[0]
            penalties = result[1]
            if (self.verbose == True):
                print('-'*60)
                print(f'Best fitness value (with constraint) for {currentGeneration} population = {popFitnesses.max()}')
                print(f'Penalty value for best fitness = {penalties[popFitnesses.argmax()]}')
                print(f'Number of selected features = {population[popFitnesses.argmax()].sum()}')
                print('-'*60)
                print('')

            # Запомним лучшего индивида с УЧЕТОМ ДОПУСТИМОСТИ + его пригодность + его штраф
            feasible_solution = []
            feasible_fit = []
            for i in range(self.n_population):
                if ((phi.iloc[i]['phi1'] == 0) and (phi.iloc[i]['phi2'] == 0) and (phi.iloc[i]['phi3'] == 0) and (phi.iloc[i]['phi4'] == 0)):
                    feasible_solution.append([population[i],
                                              penalties[i]])
                    feasible_fit.append(popFitnesses[i])

            if (len(feasible_solution) != 0):
                results.append(np.max(feasible_fit))
                indexes_mask.append(
                    np.array(feasible_solution[np.argmax(feasible_fit)][0]) > 0)
                ind_sum.append(
                    sum(feasible_solution[np.argmax(feasible_fit)][0]))

            # замещаем старую популяцию новой
            population = self.runOneGeneration(population, self.chromosomeLength, self.n_population, popFitnesses,
                                               self.crossoverType, self.mutationProb)
        if (len(results) == 0):
            print('There is no solution satisfying conditions')
            results.append(0)
            return [], []
        else:
            # Вывод: Лучшее значение пригодности || Количество лучших признаков
            print(
                f'Best {self.scoring} score: {np.max(results)/100} || Final number of features: {ind_sum[np.argmax(results)]}')
            # Лучшее значение пригодности || Индексы лучших признаков
            return np.max(results)/100, np.arange(np.shape(self.X)[1])[indexes_mask[np.argmax(results)]]
