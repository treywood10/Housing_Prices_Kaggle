#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File for housing prices kaggle competition. 

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview
"""

# Libraries #
from math import ceil, exp, sqrt
import pandas as pd
import numpy as np
import random as ran
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as RFR
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


# Random draw of seed for random state #
seed = int(ran.uniform(1, 9999))
ran.seed(seed)


# Set mnumber of cross folds #
cv = 5


# Import training data #
train = pd.read_csv('train.csv')


# Split into X and Y #
X = train.drop(['SalePrice', 'Id'], axis = 1)
y = pd.DataFrame(train['SalePrice'])


# Find missing values #
missing = X.isnull().sum().sort_values(ascending = False)


# Make matrix to compare models #
final = pd.DataFrame(columns = ['Model', 'RMSE', 'hypers'])


################################
### Alter Based on Knowledge ###
################################

# Assign no pool (NP) to PoolQC #
X['PoolQC'].fillna('NP', inplace = True)


# Assign no feature (NF) to MiscFeature #
X['MiscFeature'].fillna('NF', inplace = True)


# Assign no ally (NAL) to Alley #
X['Alley'].fillna('NAL', inplace = True)


# Assign no fence (NF) to Fence #
X['Fence'].fillna('NF', inplace = True)


# Assign no fire place (NFP) to FireplaceQu #
X['FireplaceQu'].fillna('NFP', inplace = True)


# Assign no garage (NG) to GarageType #
X['GarageType'].fillna('NG', inplace = True)


# Fill garage variables with NG if no garage #
garage = ['GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']
for x in garage:
    X[x].fillna('NG', inplace = True)
del x, garage


# Fix GarageYrBlt since it was mixed type #
X['GarageYrBlt'] = X['GarageYrBlt'].astype(str)


# Fill basement varaibles with no basement (NB) #
basement = ['BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']
for x in basement:
    X[x].fillna('NB', inplace = True)
del x, basement


# Plot of Y #
plt.hist(y)
mean_value = y.mean()
median_value = y.median()
plt.axvline(mean_value.item(), color='red', linestyle='--', label='Mean')
plt.axvline(median_value.item(), color='blue', linestyle='--', label='Median')
plt.xlabel('Sale Price')
plt.ylabel('Density')
plt.title('Histogram of Target Variable')

# Add the text to the legend
mean_legend = plt.Line2D([], [], color='red', linestyle='--', label=f"Mean: {mean_value.item():.2f}")
median_legend = plt.Line2D([], [], color='blue', linestyle='--', label=f"Median: {median_value.item():.2f}")
plt.legend(handles=[mean_legend, median_legend])

plt.show()
del mean_value, median_value


# Natural log the target variable #
y_log = np.log(y)

plt.hist(y_log)
mean_value = y_log.mean()
median_value = y_log.median()
plt.axvline(mean_value.item(), color='red', linestyle='--', label='Mean')
plt.axvline(median_value.item(), color='blue', linestyle='--', label='Median')
plt.xlabel('Sale Price')
plt.ylabel('Density')
plt.title('Histogram of Target Variable')

# Add the text to the legend
mean_legend = plt.Line2D([], [], color='red', linestyle='--', label=f"Mean: {mean_value.item():.2f}")
median_legend = plt.Line2D([], [], color='blue', linestyle='--', label=f"Median: {median_value.item():.2f}")
plt.legend(handles=[mean_legend, median_legend])

plt.show()
del mean_legend, mean_value, median_legend, median_value


#######################
#### Preprocessing ####
#######################

# Mask of numerical features #
numeric_feats = X.select_dtypes(include = ['int64', 'float64']).columns


# Mask categorical features #
cat_feats = X.select_dtypes(include = ['object']).columns


# Encode object variables #
ord_enc = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=np.nan)
X[cat_feats] = ord_enc.fit_transform(X[cat_feats])


# KNN Imputer #
knn_im = KNNImputer(n_neighbors = 10, weights = 'distance')
X_imp = pd.DataFrame(knn_im.fit_transform(X), columns = X.columns)


# Variables need to be made integer #
X_imp[['Electrical', 'MasVnrType']] = X_imp[['Electrical', 'MasVnrType']]\
                                        .apply(lambda x: x.apply(ceil))


##########################
#### Data Exploration ####
##########################

# Correlation of numeric feats and target #
corr_val = pd.concat([X_imp[numeric_feats], y], axis =1).corr()['SalePrice']


# Mask of features with high correlation to target #
select_num = list(corr_val[(corr_val >= 0.50) | (corr_val <= -0.50)].index)
select_num.remove('SalePrice')


# Get mutual information of categorical features and outcome #
fs = SelectKBest(score_func = mutual_info_regression, k = 'all')
fs.fit(X_imp[cat_feats], np.ravel(y_log))


# Sort the scores and corresponding feature names in descending order
sorted_scores, sorted_features = zip(*sorted(zip(fs.scores_,
                                                 X_imp[cat_feats].columns),
                                             reverse=True))


# Select the top 20 features
top_20_cat_features = list(sorted_features[:20])


# Combine feature selection list #
selected_feats = select_num + top_20_cat_features


# DF of only selected features #
X_imp = X_imp[selected_feats]
del corr_val, fs, sorted_features, sorted_scores


########################
#### Transformation ####
########################

# Categorical transformer #
cat_trans = Pipeline(steps = [
    ('encode', OneHotEncoder(sparse_output = False))])


# Numeric tranformer #
num_trans = Pipeline(steps = [
    ('stand', StandardScaler())])


# Construct processor #
processor = ColumnTransformer(
    transformers = [
        ('num', num_trans, select_num),
        ('cat', cat_trans, top_20_cat_features)],
    remainder = 'passthrough')


# Apply processor #
temp = processor.fit_transform(X_imp)


# Get categorical feature names #
enc_cat_features = list(processor.named_transformers_['cat']['encode']\
                        .get_feature_names_out())


# Concat label names #
labels = select_num + enc_cat_features 


# Make df of processed data #
X_train = pd.DataFrame(temp, columns = labels)
del missing, temp, train, X, X_imp


###############
#### Ridge ####
###############

# Define objective for ridge #
def obj_ridge(alpha, fit_intercept, solver):
    
    """
    Objective function to minimize the error of the 
    ridge regression.    

    Parameters
    ----------
    alpha : L2 Regularization term.
        Regularizes the coefficients. Values stipulated
        in pbounds.
    fit_intercept : Boolean of fit intercept.
        Indicator of whether or not the model
        fits an intercept.
    solver : Solving method of ridge regression.
        Continuous variable for selecting the best
        solver for the regression.

    Returns
    -------
    error : Mean squared error.
        Cross validation returns root mean error that is later
        convereted into RMSE in the comparison frame.

    """
    
    # Fit intercept #
    fit_intercept = bool(round(fit_intercept))
    
    # Solver #
    if solver <= 1.0:
        solver = 'auto'
    elif solver <= 2.0:
        solver = 'svd'
    elif solver <= 3.0:
        solver = 'cholesky'
    elif solver <= 4.0:
        solver = 'lsqr'
    elif solver <= 5.0:
        solver = 'sparse_cg'
    elif solver <= 6.0:
        solver = 'sag'
    else:
        solver = 'saga'
    
    # Instantiate ridge model #
    model = Ridge(alpha=alpha, fit_intercept=fit_intercept, solver=solver,
                  max_iter = 20000)
    
    # Cross validation and mean MSE #
    error = cross_val_score(model, X_train, y_log, cv=cv,
                            scoring='neg_mean_squared_error').mean()
    
    # Return error #
    return error


# Define search space #
pbounds = {
    'alpha': (0.0000001, 100),
    'fit_intercept': (0, 1),
    'solver': (0, 8),
}

# Set the optimizer #
optimizer = BayesianOptimization(
    f=obj_ridge, pbounds=pbounds, random_state=seed)

# Call maximizer #
optimizer.maximize(init_points=50, n_iter=450)


# Pull best info #
best_hypers = optimizer.max['params']
best_mse = optimizer.max['target']


# Fill comparison matrix #
final = final.append(
    {'Model' : 'Ridge',
     'RMSE': np.sqrt(best_mse * -1),
     'hypers': best_hypers},
    ignore_index = True
   )
final = final.sort_values('RMSE')


###############
#### Lasso ####
###############

# Define objective function for lasso #
def obj_lasso(alpha, fit_intercept,
              selection): 
    
    """
    The objective of this function is to minimize the error
    of the lasso function. 
    
    Parameters
    ----------
    alpha : L1 Regularization term.
        Regularizes the coefficients. Values stipulated
        in pbounds.
    fit_intercept : Boolean of fit intercept.
        Indicator of whether or not the model
        fits an intercept.
    selection : Dictates coefficient updates.
        Continuous variable of using either cycle or
        random for coefficient update.

    Returns
    -------
    error : Mean squared error.
        Cross validation returns root mean error that is later
        convereted into RMSE in the comparison frame.
    """
    
    # Fit intercept #
    fit_intercept = bool(round(fit_intercept))
    
    
    # selection #
    if selection <= 0.5:
        selection = 'cyclic'
    else:
        selection = 'random'
    
    # Instantiate model #
    model = Lasso(alpha = alpha, fit_intercept = fit_intercept,
                  selection = selection,
                  random_state = seed, max_iter = 20000)
    
    # Cross validation and mean MSE #
    error = cross_val_score(model, X_train, y_log, cv=cv,
                            scoring='neg_mean_squared_error').mean()
    
    # Return error #
    return error


# Define search space #
pbounds = {
    'alpha': (0.0000001, 100),
    'fit_intercept': (0, 1),
    'selection': (0, 1)
}


# Set the optimizer #
optimizer = BayesianOptimization(
    f=obj_lasso, pbounds=pbounds, random_state=seed)

# Call maximizer #
optimizer.maximize(init_points = 50, n_iter = 450)


# Pull best info #
best_hypers = optimizer.max['params']
best_mse = optimizer.max['target']


# Fill comparison matrix #
final = final.append(
    {'Model' : 'Lasso',
     'RMSE': np.sqrt(best_mse * -1),
     'hypers': best_hypers},
    ignore_index = True
   )
final = final.sort_values('RMSE')


################################
#### Elastic Net Regression ####
################################

# Define objective function for Net #
def obj_net(alpha, l1_ratio, fit_intercept,
            selection):
    
    # Vary fit intercept #
    fit_intercept = bool(round(fit_intercept))

    # Vary selection #
    if selection <= 0.5:
        selection = 'cyclic'
    else:
        selection = 'random'
        
    # Instantiate the model #
    model = ElasticNet(alpha = alpha, l1_ratio = l1_ratio,
                       fit_intercept =  fit_intercept,
                       selection = selection, random_state = seed,
                       max_iter = 20000)
    
    # Cross validation and mean MSE #
    error = cross_val_score(model, X_train, np.ravel(y_log), cv=cv,
                            scoring='neg_mean_squared_error').mean()
    
    # Return error #
    return error

# Define search space #
pbounds = {
    'alpha': (0.00001, 5),
    'l1_ratio': (0.0001, 0.999),
    'fit_intercept': (0, 1),
    'selection': (0, 1)
}   


# Set the optimizer #
optimizer = BayesianOptimization(
    f=obj_net, pbounds=pbounds, random_state=seed)


# Call maximizer #
optimizer.maximize(init_points = 50, n_iter = 450)


# Pull best info #
best_hypers = optimizer.max['params']
best_mse = optimizer.max['target']


# Fill comparison matrix #
final = final.append(
    {'Model' : 'ElasticNet',
     'RMSE': np.sqrt(best_mse * -1),
     'hypers': best_hypers},
    ignore_index = True
   )
final = final.sort_values('RMSE') 


###################################
#### Support Vector Regression ####
###################################

# Define objective function for SVM #
def obj_SVR(kernel, degree,
            gamma, C, epsilon, 
            shrinking):
    """
    

    Parameters
    ----------
    kernel : Kernel used in solver.
        String inputs that are used in optimizer.
    degree : Degree of polyomial kernel.
        Only used in poly alogrithm.
    gamma : Kernel cofficient.
        Only used in rbf, poly, and sigmoid.
    C : L2 regularizer.
        More regularization at smaller values.
    epsilon : Epplison value in SVR model.
        Specifies penalty in training loss function.
    shrinking : Boolean value.
        Dictates if the model uses shrinking heuristic.

    Returns
    -------
    error : Mean squared error.
        Cross validation returns root mean error that is later
        convereted into RMSE in the comparison frame.

    """
    
    # Kernel #
    if kernel <= 1:
        kernel = 'linear'
    elif kernel <= 2:
        kernel = 'poly'
    elif kernel <= 3:
        kernel = 'rbf'
    else:
        kernel = 'sigmoid'
    
    # Gamma #
    if gamma <= 0.5:
        gamma = 'scale'
    else:
        gamma = 'auto'
        
    # Shrinking #
    shrinking = bool(round(shrinking))
        
    # Instantiate SVR #
    model = SVR(kernel =  kernel, degree = int(degree),
                gamma = gamma, C = C,
                epsilon = epsilon, shrinking = shrinking,
                max_iter = 50000)
    
    # Cross validation and mean MSE #
    error = cross_val_score(model, X_train, np.ravel(y_log), cv=cv,
                            scoring='neg_mean_squared_error').mean()
    
    # Return error #
    return error
    

# Define search space #
pbounds = {
    'kernel': (0, 4),
    'degree': (1, 10),
    'gamma': (0, 1),
    'C': (0.0001, 100),
    'epsilon': (0.0001, 100),
    'shrinking': (0, 1)
}


# Set the optimizer #
optimizer = BayesianOptimization(
    f=obj_SVR, pbounds=pbounds, random_state=seed)


# Call maximizer #
optimizer.maximize(init_points = 50, n_iter = 450)


# Pull best info #
best_hypers = optimizer.max['params']
best_mse = optimizer.max['target']


# Fill comparison matrix #
final = final.append(
    {'Model' : 'SVR',
     'RMSE': np.sqrt(best_mse * -1),
     'hypers': best_hypers},
    ignore_index = True
   )
final = final.sort_values('RMSE')


########################
#### Random Forrest ####
########################

# Define objective function for random forest #
def obj_RF(n_estimators, criterion,
           min_samples_split, min_samples_leaf,
           max_features, bootstrap, min_impurity_decrease):
    """
    

    Parameters
    ----------
    n_estimators : Float
        Number of trees to estimate in the forest.
    criterion : String
        How the tree measures quality of the split.
    min_samples_split : Float
        Minimum number of samples required to split a node.
    min_samples_leaf : Float
        Minimum number of samples required to be a leaf.
    max_features : String
        Number of features to consider when splitting.
    bootstrap : Boolean
        Whether bootstraps are used when building trees.
    min_impurity_decrease : Float
        Node is split if it decreases the impurity.

    Returns
    -------
    error : Mean squared error.
        Cross validation returns root mean error that is later
        convereted into RMSE in the comparison frame.

    """
    
    # Criterion #
    if criterion <= 1.0:
        criterion = 'squared_error'
    elif criterion <= 2.0:
        criterion = 'absolute_error'
    elif criterion <= 3.0:
        criterion = 'friedman_mse'
    else:
        criterion = 'poisson'
        
    # Max features #
    if max_features <= 0.5:
        max_features = 'sqrt'
    else:
        max_features = 'log2'
        
    # Bootstrap #
    bootstrap = bool(round(bootstrap))
    
    # instantiate random forest moel #
    model = RFR(n_estimators = int(n_estimators), criterion = criterion,
                min_samples_split =  min_samples_split,
                min_samples_leaf = min_samples_leaf,
                max_features =  max_features, bootstrap =  bootstrap,
                min_impurity_decrease = min_impurity_decrease,
                n_jobs = -1, random_state = seed)
    
    # Cross validation and mean MSE #
    error = cross_val_score(model, X_train, np.ravel(y_log), cv=cv,
                            scoring='neg_mean_squared_error').mean()
    
    # Return error #
    return error


# Define search space #
pbounds = {
    'n_estimators': (1, 1000),
    'criterion': (0, 4),
    'min_samples_split': (0.01, .90),
    'min_samples_leaf': (0.01, .90),
    'max_features': (0, 1),
    'bootstrap': (0, 1),
    'min_impurity_decrease': (0.001, 0.4)
}


# Set the optimizer #
optimizer = BayesianOptimization(
    f=obj_RF, pbounds=pbounds, random_state=seed)


# Call maximizer #
optimizer.maximize(init_points = 50, n_iter = 450)


# Pull best info #
best_hypers = optimizer.max['params']
best_mse = optimizer.max['target']


# Fill comparison matrix #
final = final.append(
    {'Model' : 'Random Forest',
     'RMSE': np.sqrt(best_mse * -1),
     'hypers': best_hypers},
    ignore_index = True
   )
final = final.sort_values('RMSE')


############################
#### XGBoost Regression ####
############################

# Define objective function for XGBoost regression #
def obj_boost(n_estimators, eta, gamma, 
              max_depth, subsample, colsample_bytree,
              reg_lambda, alpha):
    
    
    
    # instantiate XGBoost #
    model = XGBRegressor(n_estimators = int(n_estimators), eta = eta,
                         gamma = gamma, max_depth = int(max_depth),
                         subsample = subsample, colsample_bytree = colsample_bytree,
                         reg_lambda = reg_lambda, alpha = alpha, 
                         seed = seed, n_jobs = -1)
    
    # Cross validation and mean MSE #
    error = cross_val_score(model, X_train, np.ravel(y_log), cv=cv,
                            scoring='neg_mean_squared_error').mean()
    
    # Return error #
    return error

    
# Define the search space #
pbounds = {
    'n_estimators': (1, 2000),
    'eta': (0, 1),
    'gamma': (0, 5),
    'max_depth': (2, 10),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.2, 1),
    'reg_lambda': (0, 10),
    'alpha': (0, 10)
}


# Set the optimizer #
optimizer = BayesianOptimization(
    f=obj_boost, pbounds=pbounds, random_state=seed)


# Call maximizer #
optimizer.maximize(init_points = 50, n_iter = 450)


# Pull best info #
best_hypers = optimizer.max['params']
best_mse = optimizer.max['target']


# Fill comparison matrix #
final = final.append(
    {'Model' : 'XGBoost Reg',
     'RMSE': np.sqrt(best_mse * -1),
     'hypers': best_hypers},
    ignore_index = True
   )
final = final.sort_values('RMSE')


#############################
#### K-Nearest Neighbors ####
#############################

# Define objective function for K-Nearest Neighbors #
def obj_knn(n_neighbors, weights, algorithm,
            leaf_size, p):
    
    # Variation on weights #
    if weights <= 0.5:
        weights = 'uniform'
    else:
        weights = 'distance'
    
    # Variation on algorithm #
    if algorithm <= 1.0:
        algorithm = 'auto'
    elif algorithm <= 2.0:
        algorithm = 'ball_tree'
    elif algorithm <= 3.0:
        algorithm = 'kd_tree'
    else:
        algorithm = 'brute'
    
    # Variation on p #
    if p <= 1.0:
        p = 1
    else:
        p = 2
    
    # Instantiate model #
    model = KNeighborsRegressor(n_neighbors =  int(n_neighbors), weights = weights,
                                algorithm = algorithm, leaf_size = int(leaf_size), p = p)
    
    # Cross validation and mean MSE #
    error = cross_val_score(model, X_train, np.ravel(y_log), cv=cv,
                            scoring='neg_mean_squared_error').mean()
    
    # Return error #
    return error
    

# Define search space #
pbounds = {
    'n_neighbors': (2, 20),
    'weights': (0, 1),
    'algorithm': (0, 4),
    'leaf_size': (2, 50),
    'p': (0, 2)
}


# Set the optimizer #
optimizer = BayesianOptimization(
    f=obj_knn, pbounds=pbounds, random_state=seed)


# Call maximizer #
optimizer.maximize(init_points = 50, n_iter = 450)


# Pull best info #
best_hypers = optimizer.max['params']
best_mse = optimizer.max['target']


# Fill comparison matrix #
final = final.append(
    {'Model' : 'KNN Reg',
     'RMSE': np.sqrt(best_mse * -1),
     'hypers': best_hypers},
    ignore_index = True
   )
final = final.sort_values('RMSE')


########################
#### Neural Network ####
########################

# Define objective function for network #
def obj_net(batch_size, epochs, optimizer,
            learning_rate, activation, num_nodes,
            num_hidden_layers):
    

    # Set optimizer #
    if optimizer <= 0.25:
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        
    elif optimizer <= 0.50:
        optimizer = optimizers.RMSprop(learning_rate=learning_rate)
        
    elif optimizer <= 0.75:
        optimizer = optimizers.SGD(learning_rate = learning_rate)
        
    else:
        optimizer = optimizers.Adagrad(learning_rate=learning_rate)
        
    # Set activation function #
    if activation <= 0.25:
        activation = 'relu'
        
    elif activation <= 0.5:
       activation = 'sigmoid'
       
    elif  activation <= 0.75:
       activation = 'tanh'
       
    else:
       activation = 'elu'
       
    # Instantiate model
    model = Sequential()
    
    # Set input layer #
    model.add(Dense(int(num_nodes), activation = activation, 
                    input_shape = (X_train.shape[1],)))
    
    # Set hidden layer with batch normalizer #
    for _ in range(int(num_hidden_layers)):
        model.add(Dense(int(num_nodes), activation = activation))
        model.add(BatchNormalization())
    
    # Add output layer #
    model.add(Dense(1))
    
    # Set compiler #
    model.compile(optimizer = optimizer, loss = 'mean_squared_error')
    
    reg = KerasRegressor(build_fn = lambda: model, 
                         batch_size = int(batch_size),
                         epochs = int(epochs))

    # Cross validation and mean MSE #
    error = cross_val_score(reg, X_train, np.ravel(y_log), cv=cv,
                        scoring='neg_mean_squared_error').mean()

    # Return error #
    return error

# Define search space #
pbounds = {
    'batch_size': (73, 1460),
    'epochs': (5, 100),
    'optimizer': (0, 1),
    'learning_rate': (0.000001, 0.5),
    'num_nodes': (1, 200),
    'num_hidden_layers': (1, 100),
    'activation': (0, 1)
}


# Set the optimizer #
optimizer = BayesianOptimization(f=obj_net, pbounds=pbounds,
                                 random_state=1)


# Call the maximizer #
optimizer.maximize(init_points=50, n_iter=450)


# Pull best info #
best_hyperparameters = optimizer.max['params']
best_accuracy = optimizer.max['target']


# Fill comparison matrix #
final = final.append(
    {'Model' : 'Neural Net',
     'RMSE': np.sqrt(best_mse * -1),
     'hypers': best_hypers},
    ignore_index = True
   )
final = final.sort_values('RMSE')

