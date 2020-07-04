# SOCR - Heights and Weights Project
# Objective is to develop a predictor to predict weights as function of height using
# different regression models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.metrics import r2_score


# Load data in pandas and check for missing data
def loadData(dataFile, fileExtension='.csv'):
    # Load data from file as DataFrame
    data = None  # initialize data
    if fileExtension == '.csv':
        data = pd.read_csv(dataFile)
    elif fileExtension == '.xlsx':
        data = pd.read_excel(dataFile)

    # Peek into data set
    print(data.head(10))
    print('\n')

    # Look at information in data set
    print(data.info())
    print('\n')

    # Look at information in data set
    print(data.describe())
    print('\n')

    # Check for missing data, if any
    if not (data.isna().any().any()):
        print('There is no missing data!')
    else:
        print('Check for missing data!')

    return data

# Plot scatter plot of data
def plotData(XFeat, yTarget, yPredictor):
    plt.xlabel('Height [inches]')
    plt.ylabel('Weight [pounds]')
    plt.scatter(XFeat, yTarget, c='steelblue', edgecolor='white', s=70)
    plt.plot(XFeat, yPredictor, color='black', lw=2)
    plt.show()
    return None

# Main program
if __name__ == '__main__':
    # Load full data set
    fileName = 'heights_weights.xlsx'
    dataFull = loadData(fileName, '.xlsx')
    print('\n')

    # Extract columns from data set
    columnNames = dataFull.columns
    columnNames = [columnNames[1], columnNames[2]]

    # Plot data
    scatterplotmatrix(dataFull[columnNames].values, figsize=(10, 9),
                      names=columnNames, alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Specify feature and target variables
    X = dataFull[columnNames[0]].values  # height
    y = dataFull[columnNames[1]].values  # weight

    # Split data into training (80%)/test data (20%) sets
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train[:, np.newaxis]
    y_train = y_train[:, np.newaxis]
    X_test = X_test[:, np.newaxis]
    y_test = y_test[:, np.newaxis]

    # Scale training data
    scale = StandardScaler()
    X_train_std = scale.fit_transform(X_train)
    y_train_std = scale.fit_transform(y_train)

    # Scale test data
    X_test_std = scale.fit_transform(X_test)
    y_test_std = scale.fit_transform(y_test)

    # Choose the best model for training data
    estimators = {
        'LinearRegression': LinearRegression(),
        'ElasticNet': ElasticNet(),
        'Lasso': Lasso(),
        'Ridge': Ridge()
    }

    for estimator_name, estimator_object in estimators.items():
        kFold = KFold(n_splits=10, random_state=11, shuffle=True)
        scores = cross_val_score(estimator=estimator_object, X=X_train_std,
                                 y=y_train_std, cv=kFold, scoring='r2')
        print(f'{estimator_name:>16}: ' +
              f'mean of r2 scores={scores.mean():.3f}')

    # Fit training data to linear model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_std, y_train_std)
    print(f'Slope: {lin_reg.coef_[0][0]:.4f}')
    print(f'Intercept: {lin_reg.intercept_[0]:.4f}')

    # Generate linear fit predictor using test data
    predictor_linReg = lin_reg.predict(X_test_std)

    # Compute errors from linear regression model
    r2_val = r2_score(X_test_std, predictor_linReg)
    print(f'R^2 score:  {r2_val:.4f}')
    print('\n')

    # Plot normalized data with predictor
    plotData(X_test_std, y_test_std, predictor_linReg)

    # Apply Ridge model
    ridge_reg = Ridge(alpha=1, solver='cholesky')
    ridge_reg.fit(X_train_std, y_train_std)
    predictor_ridgeReg = ridge_reg.predict(X_test_std)
    r2_val = r2_score(X_test_std, predictor_ridgeReg)
    print(f'R^2 score:  {r2_val:.4f}')
    print('\n')



