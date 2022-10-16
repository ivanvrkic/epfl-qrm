import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal, norm, binom
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import os
from numpy.linalg import inv
import scipy as sp
from scipy import stats
import pickle
from ipdb import set_trace


class Backtest:
    def __init__(self, df_prices, window=250, shrinkage=0.5, alpha=[0.95, 0.99]):
        self.df_prices = df_prices
        self.WINDOW_SIZE = window
        self.SHRINKAGE_DEGREE = shrinkage
        self.alpha_list = alpha
        self.__load()

    def __load(self):
        """Load and prepare data."""
        # Preprocess prices DataFrame
        self.df_prices.time = pd.to_datetime(self.df_prices.time)
        self.df_prices.index = self.df_prices.time
        self.df_prices.drop("time", axis=1, inplace=True)
        # Create daily log returns DataFrame
        self.df_log_returns = np.log(self.df_prices.astype(float).pct_change()+1)\
            .iloc[1:, :].fillna(0)
        # Create losses DataFrame
        self.df_loss = self.df_prices.shift(1).iloc[1:, :] *\
            (np.exp(self.df_log_returns)-1)
        self.df_loss = self.df_loss.fillna(0)
        # Create a random vector of shares with value [1,100]
        self.shares = np.ceil(np.random.rand(self.df_log_returns.shape[1])*100)
        # Create daily portfolio losses DataFrame
        self.daily_loss = self.df_loss @ self.shares
        # Create a rolling window mean and covariance
        if os.path.isfile('./mu.pkl') and os.path.isfile('./cov.pkl'):
            self.mu_df = pd.read_pickle('./mu.pkl')
            self.cov_df = pd.read_pickle('./cov.pkl')
        else:
            self.mu_df = self.df_log_returns.rolling(
                window=self.WINDOW_SIZE).mean().dropna()
            self.cov_df = self.df_log_returns.rolling(
                window=self.WINDOW_SIZE).cov().dropna()
        # Create a separate vector which stores rolling window dates.
        # VaR will be estimated on these dates
        self.rolling_window_dates = self.mu_df.index

    def shrink(self, cov_mat, shrinkage=0.5):
        """Shrinkage function. Details: L3 Slide 32,49"""
        D_cov_mat = np.diag(np.diag(np.sqrt(cov_mat)))
        corr_mat = inv(D_cov_mat) @ cov_mat @ inv(D_cov_mat)
        shrunk_corr_mat = (1 - shrinkage) * corr_mat +\
            shrinkage * np.identity(corr_mat.shape[0])
        shrunk_cov_mat = D_cov_mat @ shrunk_corr_mat @ D_cov_mat
        return shrunk_cov_mat

    def var_mvn(self, V, mu, returnscm, alpha):
        """Function for estimating VaR with Multi-Variate Normal. Details: L3 Slide 48"""
        mu_loss = np.dot(V.values, mu)
        Vscm = V.values.reshape(1, -1) @ returnscm @ V.values.reshape(-1, 1)
        Vscm = Vscm.to_numpy().item()
        var = mu_loss + np.sqrt(Vscm) * norm.ppf(alpha)
        return var
    def backtest(self):
        """Backtesting function."""
        if os.path.isfile('./varlist.pkl'):
            self.df_VaR = pd.read_pickle('./varlist.pkl')
        else:
            # Iterate through rolling windows
            VaR_list = []
            for i, d in enumerate(self.rolling_window_dates):
                cov_i = self.cov_df.loc[d]
                # Create a covariance mask which contains only non-zero rows and columns.
                # Used to avoid singular matrices as they do not allow for matrix shrinkage
                mask_cov = ~(cov_i == 0).all()
                V = (self.df_prices.loc[d, :] * self.shares).loc[mask_cov]
                # Create a portfolio mask which contains non-NaN values 
                # which might've been caused by non-existent ticker (at selected rolling window time)
                mask_V = ~np.isnan(V)

                #Apply masks
                V = V[mask_V]
                cov_i = cov_i.loc[mask_cov, mask_cov]
                cov_i = cov_i.loc[mask_V, mask_V]
                mu_i = self.mu_df.loc[d, mask_cov & mask_V]

                # Shrink covariance matrix using shrink function
                scm_i = self.shrink(cov_i)
                # Estimate VaR with MVN distribution
                var = self.var_mvn(V, mu_i, scm_i, self.alpha_list)

                VaR_list.append(var)
            self.df_VaR = pd.DataFrame(VaR_list, columns=self.alpha_list)
            self.df_VaR.index = self.rolling_window_dates
        # Create a boolean DataFrame with breaches
        self.breaches_logic = pd.DataFrame([self.daily_loss[self.rolling_window_dates]
                                            > self.df_VaR[a] for a in self.alpha_list]).T
        self.breaches_logic.columns = self.alpha_list

    def get_probability(self):
        """Estimate probability of breaches."""
        self.number_of_breaches = self.breaches_logic.sum()

        for alpha in self.alpha_list:
            no_b = self.number_of_breaches[alpha]
            n = len(self.rolling_window_dates)
            prob = binom.sf(k=no_b - 1, n=n, p=1 - alpha)
            print(f"(alpha={alpha}) No. breaches: {no_b}\n"
                  f"(alpha={alpha}) Probability of atleast {no_b} breaches "
                  f"(i.e. 1-Bin({no_b-1},{n},{1-alpha:.2f}): {prob:.4%}")

    def plot_breaches(self, save=True):
        """Plot breaches."""
        self.breaches = self.df_VaR[self.breaches_logic]
        c_var = ['orange', 'cyan']
        c_b = ['r', 'b']
        plt.figure(figsize=(13, 10))
        plt.plot(self.daily_loss[self.rolling_window_dates],
                 label='Daily Loss',
                 color='k',
                 linewidth=0.5,
                 zorder=1)
        for i, alpha in enumerate(self.alpha_list):
            breaches_alpha = self.breaches[alpha]
            breaches_alpha.dropna(inplace=True)
            plt.scatter(breaches_alpha.index,
                        breaches_alpha.values,
                        s=5,
                        color=c_b[i],
                        zorder=3)
            plt.plot(self.df_VaR[alpha],
                     label=r'$VaR, \alpha=$'+str(alpha),
                     color=c_var[i],
                     linewidth=0.8,
                     zorder=2)
        plt.xlabel('Date')
        plt.ylabel('Daily Loss')
        plt.legend()
        if save:
            plt.savefig('fig.png')
            plt.savefig('fig.pdf')
        plt.show()


np.random.seed(1)

df_prices = pd.read_csv('stock_prices_SP500_2007_2021.csv')
backtest = Backtest(df_prices)
backtest.backtest()
backtest.get_probability()
backtest.plot_breaches()
