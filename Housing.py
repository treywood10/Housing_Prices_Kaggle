#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File for housing prices kaggle competition. 

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview
"""

# Libraries #
from math import ceil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import Ridge, Lasso
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score


# Import training data #
train = pd.read_csv('train.csv')


# Split into X and Y #
X = train.drop(['SalePrice', 'Id'], axis = 1)
y = pd.DataFrame(train['SalePrice'])


# Find missing values #
missing = X.isnull().sum().sort_values(ascending = False)


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
fs = SelectKBest(score_func = mutual_info_classif, k = 'all')
fs.fit(X_imp[cat_feats], np.ravel(y))


# Sort the scores and corresponding feature names in descending order
sorted_scores, sorted_features = zip(*sorted(zip(fs.scores_, X_imp[cat_feats].columns), reverse=True))


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



