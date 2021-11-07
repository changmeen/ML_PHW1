# ML_PHW1  
Gachon Univ. SW 201735826 류창민
## Best_Model
__Best_Model__ is the main function

__Best_Model__ is __Semi-AUTO ML__ function


You have to put dataframe(After handling noise data), target column name as input of Best_Model function

__variables in Best_Model__

X - Predictor variables

y - target variable

X_cate - predictor variables that are Categorical values

X_nume - predictor variables that are Numeric values

df_cate_empty - boolean checker for if X_cate is empty or not

df_nume_empty - boolean checker for if X_nume is empty or not

In this function we will use 4 scalers - Standard Scaler, MinMaxScaler, MaxAbsScaler, RobustScaler

In this function we will use 2 encoders - Ordinal Encoder, OneHot Encoder

In this function we will use 4 Classifiers - DecisionTree Classifier(gini),  
DecisionTree Classifier(entropy), Logistic Regression, Support Vector Machine(SVM)
