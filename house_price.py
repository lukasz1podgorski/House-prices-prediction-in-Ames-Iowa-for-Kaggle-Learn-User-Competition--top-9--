import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Read the data
X_full = pd.read_csv('train.csv', index_col='Id')
X_test_full = pd.read_csv('test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y,
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and
                    X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if
                X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

numerical_transformer = SimpleImputer(strategy='median')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

max_depths = np.linspace(1, 32, 32, endpoint=True, dtype='int16')

####################################################
# Function to get MAE for model tuning
def get_mae(n_estimators, X_train, X_valid, y_train, y_valid):
    gb_model = XGBRegressor(random_state=0, n_estimators=n_estimators, learning_rate=0.18, n_jobs=4)

    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', gb_model)
                             ])

    # Preprocessing of training data, fit model
    my_pipeline.fit(X_train, y_train)

    preds = my_pipeline.predict(X_valid)
    score = mean_absolute_error(y_valid, preds)
    return(score)

####################################################

# XGBOOST
gb_model = XGBRegressor(random_state=42, n_estimators=1700, learning_rate=0.06, n_jobs=4)

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', gb_model)
                             ])

# Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)

preds = my_pipeline.predict(X_valid)
score = mean_absolute_error(y_valid, preds)

print('MAE:', score)

# Looping through candidates for parameters
#for n_estimators in [1600, 1630, 1650, 1680, 1700, 1710, 1720, 1750]:
#    my_mae = get_mae(n_estimators, X_train, X_valid, y_train, y_valid)
#    print("estimators: %d  \t\t Mean Absolute Error:  %d" %(n_estimators, my_mae))

# Prediction
preds_test = my_pipeline.predict(X_test)

# Save output to csv file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)