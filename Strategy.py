from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

"""
Model Part
Include the alpha predictive model, rolling fitting model and the portfolio optimization model
"""


# core of the factor combination model

# core of regression model
class ridgeModel:
    # ridge regression
    def __init__(self):
        self.model = None

    def fit(self, x, y, penalty=1):
        # both x and y should be numpy matrices
        self.model = Ridge(fit_intercept=False, alpha=penalty)
        self.model.fit(x, y)
        self.coef = self.model.coef_


# core of gbdt model
class gBDTModel:
    def __init__(self):
        self.model = None

    def fit(self, x, y, n_estimators=100, max_depth=1):
        self.model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth)
        self.model.fit(x, y)


def mean_variance_optimization(mean_pnl, cov_pnl):
    """
    implement mean variance optimization strategy
    :param mean_pnl: mean pnl
    :param cov_pnl: cov pnl
    :return: estimated weight
    """
    cov_pnl_inv = np.linalg.inv(cov_pnl)

    # weight matrix
    w = cov_pnl_inv @ mean_pnl + (1 - np.sum(cov_pnl_inv @ mean_pnl)) / (np.sum(cov_pnl_inv))
    # in case of numeric error
    w = np.where(w < 0, 0, w)
    w = w / np.sum(w)
    return w


class rollingStrategy:
    def __init__(self, cooldown=365, lookback=4 * 365, retrain_window=365):
        """
        initialize the rolling fitting strategy
        :param cooldown: how long does a specific strategy cools down
        :param lookback: size of the training set
        :param retrain_window: how long do we re train the model
        """
        self.cooldown = cooldown
        self.lookback = lookback
        self.retrain_window = 365
        self.data = None
        self.model = None

    def roll(self, data, model, signal_names, **params):
        # Sort the data by date
        # data = data.sort_values('date')

        # Initialize the model with the given parameters
        base_model = model()

        self.data = data
        self.model = model
        self.signal_names = signal_names

        # Convert the 'date' column to datetime if it's not already
        data['date'] = pd.to_datetime(data['date'])

        # Create a new column for predictions
        data['predictions'] = None

        # Group the data by ticker and apply the rolling window strategy to each group
        for ticker, ticker_data in data.groupby('ticker'):
            # Initialize a separate model for this ticker
            model = self.model()

            min_date = ticker_data['date'].min()
            max_date = ticker_data['date'].max()

            current_date = min_date + timedelta(days=self.cooldown)

            while current_date < max_date:
                # Select the training and test data
                training_data = ticker_data[(ticker_data['date'] < current_date) & (
                            ticker_data['date'] >= current_date - timedelta(days=self.lookback))]
                test_data = ticker_data[(ticker_data['date'] >= current_date) & (
                            ticker_data['date'] < current_date + timedelta(self.retrain_window))]

                training_data = training_data.fillna(0)
                test_data = test_data.fillna(0)

                # Train the model on the training data
                model.fit(training_data[signal_names], training_data['daily_return'], **params)

                # Make predictions on the test data
                if test_data[signal_names].shape[0] > 0:
                    pred = model.model.predict(test_data[signal_names])
                    data.loc[test_data.index, 'predictions'] = pred
                else:
                    pass

                # Update the current date
                current_date += timedelta(days=self.retrain_window)

        self.data = data
        return data

    def portfolio_optimization(self, optimizer=mean_variance_optimization):
        """
        calculate the metrics of the dataset
        :return: return the metrics
        """
        # check if using our own data
        data = self.data
        self.optimizer = optimizer

        data = data.fillna(0)
        data['positions'] = data['predictions'] * data['clip']
        data['trades'] = data.groupby('ticker')['positions'].diff()
        data['pnl'] = data['trades'] * data['daily_return']
        # data['overall_pnl'] = data.groupby('ticker')['pnl'].sum()
        data['overall_pnl'] = None

        min_date = data['date'].min()
        max_date = data['date'].max()
        current_date = min_date + timedelta(days=self.cooldown)
        while current_date < max_date:
            training_data = data[(data['date'] < current_date) & (
                    data['date'] >= current_date - timedelta(days=self.lookback))]
            test_data = data[(data['date'] >= current_date) & (
                    data['date'] < current_date + timedelta(self.retrain_window))]

            training_data = training_data.fillna(0)
            test_data = test_data.fillna(0)

            mean_pnl = training_data.groupby('ticker')['pnl'].mean()
            df_pivot = training_data.pivot(index='date', columns='ticker', values='pnl')
            df_pivot = df_pivot
            cov_pnl = df_pivot.cov()
            cov_pnl = cov_pnl + 1 * np.eye(cov_pnl.shape[0])
            w = self.optimizer(mean_pnl, cov_pnl)

            test_data['weighted_pnl'] = test_data.groupby('ticker')['pnl'] * w
            # Group by 'time' and calculate the sum of 'weighted_pnl'
            weighted_sum_pnl = test_data.groupby('date')['weighted_pnl'].sum()

            # Optionally, you can drop the 'weighted_pnl' column if it's no longer needed
            test_data.drop(columns='weighted_pnl', inplace=True)
            print(weighted_sum_pnl)
            # calculate the reweighted overall pnl
            current_date += timedelta(days=self.retrain_window)

        return data

    def get_pnl(self, signal_name):
        # get pnl curve
        data = self.data
        data = data.fillna(0)
        data['positions'] = data[signal_name] * data['clip']
        data['trades'] = data.groupby('ticker')['positions'].diff()
        data['pnl'] = data['trades'] * data['daily_return']
        pnl = data.groupby('date')['pnl'].sum()
        return pnl
