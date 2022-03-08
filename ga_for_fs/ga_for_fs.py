#!/usr/bin/env python
from logging import raiseExceptions
import numpy as np
import pandas as pd
from pandas.core.arrays.sparse import dtype
from sklearn.model_selection import check_cv, cross_val_score
from sklearn.metrics import check_scoring
from sklearn.model_selection._validation import _fit_and_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Under Python 3.4+ use the 'forkserver' start method by default: this makes it
# possible to avoid crashing 3rd party libraries that manage an internal thread
# pool that does not tolerate forking
# os.environ['JOBLIB_START_METHOD'] = 'forkserver'
# print(f'Curent JOBLOB environment: {os.environ.get("JOBLIB_START_METHOD")}')

#  НЕ ХВАТАЕТ ПАРАМЕТРА НАСТРОЙКИ турнира - сколько брыть индивидов для турнира
class GeneticAlgorithm(object):

    def __init__(self, X, y, estimator, scoring, cv, n_population, n_gen, crossoverType, mutationProb, initType, extraFeatures_num, num_features_to_init=None, verbose=False):
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
                    Currently, only Random Forest classifier is supported!

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
            -   Uniform: "Uniform" - The decision to choose the gene whether from parent1
            or parent2 in an offspring is being made by random generator uniformly
            distributed.

        mutationProb : float
            Mutation probability for genetic algorithm. Usual practice is to set the value to: 1/n_features.

        initType: string
            Initialization type of genetic algorithm. Available options:
            – "coin" - toss a coin to decide whether to pick a feature or not (from binomial distribution),
            - "uniform_fixed_fnum" - Should be used in conjunction with "num_features_to_init"
            for datasets with large number of features (approx. > 30) to ensure convergence 
            (the ability of the algorithm to find a solution under constraints). Each feature has 
            the same probability of being selected as the other ones (from uniform distribution).

        num_features_to_init : int, default=None
            The fixed number of to choose in the initialization. 
            It is used with initType: "uniform_fixed_fnum" only.

        extraFeatures_num : int
            Number of additional (extra) features. For example, extracted
            features (PCA, kernel PCA, Autoencoder, etc.) which were added to the 
            original features set. Two of four constraints of the algorithm
            will make search withing these features.

        verbose : bool, default=False
            Regulates verbosity of the algorithm (penalty, fitness function, etc) for each epoch.
        """
        self.X = X
        self.y = y
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.n_population = n_population
        self.n_gen = n_gen
        self.chromosomeLength = X.shape[1]
        self.crossoverType = crossoverType
        self.mutationProb = mutationProb
        self.initType = initType
        self.num_features_to_init = num_features_to_init
        self.extraFeatures_num = extraFeatures_num
        self.verbose = verbose  # Для вывода подробной статистики

    def createPopulation(self, n_population, chromosomeLength, initType, num_features_to_init=None):
        "return matrix with population"
        population = []
        if (initType == 'coin'):
            # use binomial distribution
            population = np.random.binomial(
                1, 0.5, size = chromosomeLength * n_population)
            population = population.reshape(n_population, chromosomeLength)
            return population

        if (initType == 'uniform_fixed_fnum'):
            
            # if (num_features_to_init != None):
                # for k in range(n_population):
                #     # сэмплирование с повторением (имитация цикла для каждого признака) || Равномерное распр.
                #     samples = np.random.choice(chromosome_indexes,
                #                                chromosomeLength,
                #                                replace=True)
                #     genes = np.unique(samples)
                #     population[k] = np.isin(
                #         chromosome_indexes, genes, assume_unique=True)*1
            # else:
            # if num_features_to_init == None:
            #     raiseExceptions("You should pass 'num_features_to_init' to the constructor")
            # else:
            if num_features_to_init == None:
                print('')
                print("You should pass 'num_features_to_init' to the constructor")
                print('')
                return None
            else:
                population = np.zeros((n_population, chromosomeLength))
                chromosome_indexes = np.arange(chromosomeLength)
                for k in range(n_population):
                    # use uniform distribution
                    genes = np.random.choice(chromosome_indexes,
                                           num_features_to_init,
                                           replace = False)
                    # Если без повторения, то зачем искать unique?
                    # genes = np.unique(samples)
                    population[k] = np.isin(
                        chromosome_indexes, genes, assume_unique=True)*1
            return population

    def selectionTNT(self, tntSize, n_population, fitnessValues):
        "Tournament selection. Returns pair of parent indexes"
        # generate T random indexes without replacement
        indexes = np.random.choice(
            range(n_population), size=tntSize, replace=False)
        # first parent is the one with the best fitness value across 'indexes'
        parent1index = indexes[np.argmax(fitnessValues[indexes])]
        # same for the second parent
        indexes = np.random.choice(
            range(n_population), size=tntSize, replace=False)
        parent2index = indexes[np.argmax(fitnessValues[indexes])]
        return parent1index, parent2index

    def crossover(self, parent1, parent2, population, chromosomeLength, crossoverType):
        if (crossoverType == 'OnePoint'):
            # choose crossover point at random (discrete uniform distrobution)
            xpoint = np.random.randint(1, chromosomeLength)
            # blend the genes
            offspring = np.zeros(chromosomeLength)
            offspring[0:xpoint] = population[parent1, 0:xpoint]
            offspring[xpoint:] = population[parent2, xpoint:]
        if (crossoverType == 'TwoPoints'):
            # choose two crossover points at random without replacement
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
        if (y.dtype == 'int64'):
            # CV returns StratifiedkFold
            cv = check_cv(cv, y, classifier=True)
        else:
            # CV returns kFold
            cv = check_cv(cv, y)

        cv.random_state = 42
        # Shuffle is not used in order to make identical cv splits for each individual
        cv.shuffle = False
        scorer = check_scoring(estimator, scoring=scoring)
        best_params = dict()
        individual_sum = np.sum(individual, axis=0)
        if individual_sum == 0:
            scores_mean = -10000
        else:
            # Choose features according to the mask
            X_selected = X[:, np.array(individual, dtype=bool)]
            # Hyperparameters of the Random Forest classifier to adjust by Hyperopt
            tree_space = {'n_estimators' : hp.choice('n_estimators', np.arange(100, 400, 100)),
                          'max_depth' : hp.choice('max_depth', np.arange(0.1, 0.6, 0.1))}
            trials = Trials()

            def objective(params):
                clf = estimator(**params)
                f1_macro = cross_val_score(clf, X_selected, y, 
                                            scoring=scorer, 
                                            cv=cv, 
                                            n_jobs=-1).mean()

                return {'loss': -f1_macro, 'status' : STATUS_OK}

            best_params = fmin(objective, tree_space, algo=tpe.rand.suggest, max_evals=30, 
                                trials=trials, show_progressbar=False)
            scores_mean = -trials.best_trial['result']['loss']*100
        
        # Calculates extra features number in the individual
        extraFeatures = sum(individual[len(individual)-extraFeatures_num:])
        # Calculates initial (original) features number in the individual
        originalFeatures = sum(individual[:len(individual)-extraFeatures_num])

        # Penalize scores_mean according to the rules:
        phi1 = max(0, (2-extraFeatures))
        phi2 = max(0, (extraFeatures-4))
        phi3 = max(0, (2-originalFeatures))
        phi4 = max(0, (originalFeatures-4))

        phi = [phi1, phi2, phi3, phi4]

        if (verbose == True):
            print('Current cv type: ', type(cv))
            print(30*'-')
            print("Epoch = ", epoch)
            print("Individual: ", individual)
            print(f"All features (sum) {individual_sum} = constructed ({extraFeatures}) + original ({originalFeatures})")
            print("phi1 = %i, phi2 = %i, phi3 = %i, phi4 = %i:" %
                  (phi1, phi2, phi3, phi4))
            print("Objective function value = ", scores_mean)
            print('Best hyperparams found', best_params)
            print('')

        return scores_mean, phi, best_params

    def AdaptivePenalty(self, objective_func, violations):
        # fitnessFunction = np.zeros([len(objective_func), 1])
        # penalties_mas = np.zeros([len(objective_func), 1])
        fitnessFunction = np.zeros(len(objective_func))
        penalties_mas = np.zeros(len(objective_func))

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
                # Save penalty to be printed when Verbose=True
                penalties_mas[i] = penalty
        return fitnessFunction, penalties_mas

    def runOneGeneration(self, population, chromosomeLength, n_population, fitnessValues, crossoverType, mutationProb):
        newPopulation = np.zeros((n_population, chromosomeLength))
        # newPopulation = []
        for ind in range(n_population):
            # Selection
            parents = self.selectionTNT(2, n_population, fitnessValues)
            # Crossover
            offspring = self.crossover(
                parents[0], parents[1], population, chromosomeLength, crossoverType)
            # Mutation
            offspring = self.mutationPoint(offspring, mutationProb)
            # Add offspring to the population
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

        ind_best_hyperparam : dict
            Best estimator hyperparameters found during CV
            in the fitness function
        """
        results = []
        ind_sum = []
        indexes_mask = []
        best_params = []
        # Population initialization
        population = self.createPopulation(self.n_population,
                                           self.chromosomeLength,
                                           self.initType,
                                           self.num_features_to_init)
        for currentGeneration in range(self.n_gen):
            phi = pd.DataFrame(columns=['phi1', 'phi2', 'phi3', 'phi4'])
            popObjectives = np.zeros([self.n_population, 1])
            best_params_population = []
            # Calculation objective for every individual
            for ind in range(self.n_population):
                # scores_mean || phi || model.best_params_
                popObjectives[ind], phi.loc[ind], best_params_ind = self.fitness(X=self.X,
                                                                y=self.y,
                                                                estimator=self.estimator,
                                                                scoring=self.scoring,
                                                                cv=self.cv,
                                                                individual=population[ind],
                                                                epoch=currentGeneration,
                                                                extraFeatures_num=self.extraFeatures_num,
                                                                verbose=self.verbose)
                best_params_population.append(best_params_ind)

                # popObjectives[ind] = ind_fitness[0]
                # phi.loc[ind] = ind_fitness[1]
            # Impose adaptive penalty - final fitness assessement

            popFitnesses, penalties = self.AdaptivePenalty(popObjectives, phi)
            if (self.verbose == True):
                print('-'*60)
                print(f'Best fitness value (with constraint) for {currentGeneration} population = {popFitnesses.max()}')
                print(f'Penalty value for best fitness = {penalties[popFitnesses.argmax()]}')
                print(f'Number of selected features = {population[popFitnesses.argmax()].sum()}')
                print('-'*60)
                print('')

            # Save best FEASIBLE individual + its fitness + penalty + best hyperparams on CV
            feasible_solution = []
            feasible_fit = []
            feasible_best_params = []
            for i in range(self.n_population):
                if ((phi.iloc[i]['phi1'] == 0) and (phi.iloc[i]['phi2'] == 0) and (phi.iloc[i]['phi3'] == 0) and (phi.iloc[i]['phi4'] == 0)):
                    feasible_solution.append([population[i],
                                              penalties[i]])
                    feasible_fit.append(popFitnesses[i])
                    feasible_best_params.append(best_params_population[i])
            # Find the best solution within feadible individuals
            if (len(feasible_solution) != 0):
                results.append(np.max(feasible_fit))
                indexes_mask.append(
                    np.array(feasible_solution[np.argmax(feasible_fit)][0]) > 0)
                ind_sum.append(
                    sum(feasible_solution[np.argmax(feasible_fit)][0]))
                best_params.append(feasible_best_params[np.argmax(feasible_fit)])

            # Replace an old population with the new one
            population = self.runOneGeneration(population, self.chromosomeLength, self.n_population, popFitnesses,
                                               self.crossoverType, self.mutationProb)
            print(f'{currentGeneration} epoch has finished')
            
        if (len(results) == 0):
            print('There is no solution satisfying conditions')
            results.append(0)
            return [], []
        else:
            print(
                f'Best {self.scoring} score: {np.max(results)/100} || Final number of features: {ind_sum[np.argmax(results)]}')
            #        best_fitness     ||                           best_indexes                        ||        ind_best_hyperparam
            return np.max(results)/100, np.arange(np.shape(self.X)[1])[indexes_mask[np.argmax(results)]], best_params[np.argmax(results)]
