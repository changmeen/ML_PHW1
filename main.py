import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Print all the columns of dataframe
pd.set_option('display.max_columns', None)

# Read dataset and put their columns name
df = pd.read_csv('breast-cancer-wisconsin.data')
df.columns = ['id', 'thickness', 'size_uniformity',
              'shape_uniformity', 'adhesion', 'epithelial_size',
              'bare_nucleoli', 'bland_chromatin',
              'normal_nucleoli', 'mitoses', 'class']

# Print head of dataset
# print(df.head())

"""
# Find out bare_nucleoli column have '?'
c = {col:df[df[col] == "?"].shape[0] for col in df.columns}
print(c)
"""

# drop 'id' column and reset the index of dataframe
df = df.drop(['id'], axis=1)

# from 'bare_nucleoli' column drop records with '?' and reset the index
df_temp = df[df['bare_nucleoli'] == '?'].index
df = df.drop(df_temp)
df = df.reset_index(drop=True)

# This column have some problem that it only have numeric data,
# But Program recognize it as Categorical data. So, I change it's type roughly
df['bare_nucleoli'] = df['bare_nucleoli'].astype('category')


# This function can control any dataset
# Input parameter of this function
# df - input dataset
# target - target variable
def Best_Model(df, target, scalers=None, encoders=None, models=None):

    # X - predictor variables
    # y - target variable
    # X_cate - predictor variables that are Categorical values
    # X_nume - predictor variables that are Numeric values
    # df_cate_empty - boolean checker for if X_cate is empty or not
    # df_nume_empty - boolean checker for if X_nume is empty or not
    # In this function we will use 4 scalers - Standard Scaler, MinMaxScaler, MaxAbsScaler, RobustScaler
    # In this function we will use 2 encoders - Ordinal Encoder, OneHot Encoder
    # In this function we will use 4 Classifiers - DecisionTree Classifier(gini), DecisionTree Classifier(entropy),
    #                                               Logistic Regression, Support Vector Machine(SVM)

    # Split X and y
    X = df.drop(target, axis=1)
    y = df[target]

    X_cate = X.select_dtypes(include='object')
    df_cate_empty = X_cate.empty
    X_nume = X.select_dtypes(exclude='object')
    df_nume_empty = X_nume.empty

    if scalers is None:
        scale = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
    else: scale = scalers

    if encoders is None:
        encode = [OrdinalEncoder(), OneHotEncoder()]
    else: encode = encoders

    if models is None:
        model = ['DecisionTreeClassifier(gini)', 'DecisionTreeClassifier(entropy)', 'Logistic', 'SVM']
    else: model = models

    # Hyperparameter for DecisionTree Classifier(gini)
    grid_params_DT_1 = {
        'criterion': ['gini'],
        'min_samples_split': [2, 3, 4],
        'max_features': [3, 5, 7],
        'max_depth': [3, 5, 7],
        'max_leaf_nodes': list(range(7, 100))
    }

    # Hyperparameter for DecisionTree Classifier(entropy)
    grid_params_DT_2 = {
        'criterion': ['entropy'],
        'min_samples_split': [2, 3, 4],
        'max_features': [3, 5, 7],
        'max_depth': [3, 5, 7],
        'max_leaf_nodes': list(range(7, 100))
    }

    # Hyperparameter for Logistic Regression
    grid_params_LR = {
        'solver': ['lbfgs', 'liblinear'],
        'penalty': ['l2'],
        'C': np.arange(.1, 5)
    }

    # Hyperparameter for Support Vector Machine(SVM)
    grid_params_SVM = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
    }

    # These are for store best scores and parameters and scaling, encoding methods
    best_score_DT_gini = 0
    best_DT_gini_param = []
    best_en_sc_dt_gini = []
    best_cv_dt_gini = 0

    best_score_DT_entropy = 0
    best_DT_entropy_param = []
    best_en_sc_dt_entropy = []
    best_cv_dt_entropy = 0

    best_score_LR = 0
    best_LR_param = []
    best_en_sc_lr = []
    best_cv_lr = 0

    best_score_SVM = 0
    best_SVM_param = []
    best_en_sc_svm = []
    best_cv_svm = 0

    # First for loop is for various k-fold
    # k for 2 ~ 10
    for cv in range(2, 11):
        for i in scale:
            for j in encode:
                # If None of data is Numeric data, Do not scale
                if df_nume_empty is False:
                    scaler = i
                    scaler = pd.DataFrame(scaler.fit_transform(X_nume))

                # If none of data is Categorical data, Do not encode
                if j == OrdinalEncoder() and df_cate_empty is False:
                    enc = j
                    enc = enc.fit_transform(X_cate)
                    new_df = pd.concat([scaler, enc], axis=1)
                elif j == OneHotEncoder() and df_cate_empty is False:
                    dum = pd.DataFrame(pd.get_dummies(X_cate))
                    new_df = pd.concat([scaler, dum], axis=1)
                else:
                    new_df = scaler

                for k in model:
                    # Split train, test 7:3
                    X_train, X_test, y_train, y_test = train_test_split(new_df, y, test_size=0.3, random_state=42)

                    # DecisionTreeClassifier(GINI)
                    if k == 'DecisionTreeClassifier(gini)':
                        gs_DT1 = GridSearchCV(DecisionTreeClassifier(), grid_params_DT_1, verbose=1, cv=cv)
                        gs_DT1.fit(X_train, y_train)
                        score = gs_DT1.score(X_test, y_test)
                        if score > best_score_DT_gini:
                            best_score_DT_gini = score
                            best_DT_gini_param = gs_DT1.best_params_
                            best_cv_dt_gini = cv
                            best_en_sc_dt_gini = [i, j]

                    # DecisionTreeClassifier(ENTROPY)
                    if k == 'DecisionTreeClassifier(entropy)':
                        gs_DT2 = GridSearchCV(DecisionTreeClassifier(), grid_params_DT_2, verbose=1, cv=cv)
                        gs_DT2.fit(X_train, y_train)
                        score = gs_DT2.score(X_test, y_test)
                        if score > best_score_DT_entropy:
                            best_score_DT_entropy = score
                            best_DT_entropy_param = gs_DT2.best_params_
                            best_cv_dt_entropy = cv
                            best_en_sc_dt_entropy = [i, j]

                    # Logistic Regression
                    if k == 'Logistic':
                        gs_LR = GridSearchCV(LogisticRegression(), grid_params_LR, verbose=1, cv=cv)
                        gs_LR.fit(X_train, y_train)
                        score = gs_LR.score(X_test, y_test)
                        if score > best_score_LR:
                            best_score_LR = score
                            best_LR_param = gs_LR.best_params_
                            best_cv_lr = cv
                            best_en_sc_lr = [i, j]

                    # Support Vector Machine
                    if k == 'SVM':
                        gs_SVM = GridSearchCV(SVC(), grid_params_SVM, verbose=1, cv=cv)
                        gs_SVM.fit(X_train, y_train)
                        score = gs_SVM.score(X_test, y_test)
                        if score > best_score_SVM:
                            best_score_SVM = score
                            best_SVM_param = gs_SVM.best_params_
                            best_cv_svm = cv
                            best_en_sc_svm = [i, j]

    # Print results at the last of function
    # best score, best parameters, scaling, encoding method for each classifiers are posted
    print("\n")
    print("Best score for Decision Tree (GINI) is " + str(best_score_DT_gini))
    print("Model parameters: " + str(best_DT_gini_param))
    print("Best k for k-fold: " + str(best_cv_dt_gini))
    print("Scaling and Encoding Method: ", best_en_sc_dt_gini)
    print("\n")

    print("Best score for Decision Tree (ENTROPY) is " + str(best_score_DT_entropy))
    print("Model parameters: " + str(best_DT_entropy_param))
    print("Best k for k-fold: " + str(best_cv_dt_entropy))
    print("Scaling and Encoding Method: ", best_en_sc_dt_entropy)
    print("\n")

    print("Best score for Logistic Regression is " + str(best_score_LR))
    print("Model parameters: " + str(best_LR_param))
    print("Best k for k-fold: " + str(best_cv_lr))
    print("Scaling and Encoding Method: ", best_en_sc_lr)
    print("\n")

    print("Best score for SVM is " + str(best_score_SVM))
    print("Model parameters: " + str(best_SVM_param))
    print("Best k for k-fold: " + str(best_cv_svm))
    print("Scaling and Encoding Method: ", best_en_sc_svm)

    return


# Start Best_Model function
# Input is whole dataset and target variable
Best_Model(df, 'class')