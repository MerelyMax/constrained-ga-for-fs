# Feature selection using constrained genetic algorithm
**Important note: this project was build for study purposes.**

**constrained-ga-for-fs** is a Python module for automated "wrapper" feature selection. The algorithm "wraps" subset of features around a specific model (random forest was implemented so far) and uses prediction accuracy of this subset as a criterion whether to select these features or not. The algorithm was constructed with the following premise. One probably should augment original set of features with some other ones obtained by **feature reduction algorithms** (like PCA, [Autoencoder](https://deepnote.com/@maksim-denisov-c524/Autoencoder-OPTUNA-uN_zBipkSf6xsDR6-1xpVA) and others). Since feature reduction algorithms "sqeeze" the most descriminative information from the data set (descriminative in terms of classification problems), in conjunction with the original data it may probably increase predictive ability (classification accuracy). After this has been done, the best subset of features can be found within the new augmented data. Taking into account this assumption, the genetic algorithm has two constraints:
1) The subset must has from 2 to 4 features from the original data;
2) The subset must additionaly has from 2 to 4 features obtained by feature reduction algorithms (PCA, Autoencoder, e.t.c).

Therefore, the resulting subset of features is limited to the less than 8 features in total. Choice of 8 features is arbitrary and can be changed with small changes in the code.
## Wrapper feature selection technique:
![image](doc/Feature_selection_Wrapper_Method.png)
Image source: https://commons.wikimedia.org/wiki/File:Feature_selection_Wrapper_Method.png#/media/File:Feature_selection_Wrapper_Method.png

Basic idea of the "wrapper" feature selection algorithm is presented on the figure above. `Selecting the best subset` block is a genetic algorithm to `generate a subset` of features with Random Forest `learning algorithm`. Procedure can be described as follows:
1. Population initialization: each individual in the population is an array of the form - [0,0,0,1,0,1,...,1], where 1 means feature is chosen, 0 otherwise.
2. Fitness evaluation: fitness function is the Random Forest algorithm with hyperparameter search by [Hyperopt](https://github.com/hyperopt/hyperopt) and measure of the classification accuracy - `f1_macro`. Fitness function is calculated for each individual in the population.
3. Adaptive penalty: `constrained-ga-for-fs` module is based on the following hypothesis. What if 
4. 

![image](doc/GA_scheme.png)
# Install

# Example

## Attention
It tooks a lot time
