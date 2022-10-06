import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats

class DHCO:
    class CBS:
        def __init__(self, t, T, K, S, r, sigma):
            self.t = t
            self.T = T
            self.K = K
            self.S = S
            self.r = r
            self.sigma = sigma

            self.d1 = (np.log(self.S / self.K) +
                       (self.r + (self.sigma ** 2) / 2) *
                       (self.T - self.t)) /\
                      (self.sigma * np.sqrt(self.T - self.t))
            self.d2 = self.d1 - self.sigma * np.sqrt(self.T - self.t)

        def black_scholes_call(self):
            return self.S * sp.stats.norm.cdf(self.d1) -\
                   np.exp(-self.r * (self.T - self.t)) *\
                   self.K * sp.stats.norm.cdf(self.d2)

        def rfc_theta(self):
            theta = -self.S * sp.stats.norm.pdf(self.d1) *\
                    self.sigma / (2 * np.sqrt(self.T - self.t))\
                    - self.r * np.exp(-self.r * (self.T - self.t))\
                    * K * sp.stats.norm.cdf(self.d2)
            return theta

        def rfc_delta(self):
            delta = sp.stats.norm.cdf(self.d1)
            return delta

        def rfc_rho(self):
            rho = (self.T - self.t) *\
                  np.exp(-self.r *(self.T - self.t)) *\
                  self.K * sp.stats.norm.cdf(self.d2)
            return rho

        def rfc_vega(self):
            vega = self.S * sp.stats.norm.pdf(self.d1) *\
                   np.sqrt(self.T - self.t)
            return vega

    def __init__(self, t, T, K, S, r, sigma, delta, rfc_mean_vec,
                 rfc_stddev_vec, corr_mat, N):
      self.cbs = self.CBS(t, T, K, S, r, sigma)
      self.delta = delta

      self.cov_mat = np.matmul(rfc_stddev_vec.reshape(-1,1),rfc_stddev_vec.reshape(1,-1)) * corr_mat

      self.x1_delta,self.x2_delta,self.x3_delta = np.transpose(sp.stats.multivariate_normal.rvs(mean=rfc_mean_vec, cov=self.cov_mat, size=N))

      St_delta = S * np.exp(self.x1_delta)
      rt_delta = abs(r + self.x2_delta)
      sigmat_delta = abs(sigma + self.x3_delta)
      t_delta = t + delta

      self.cbs_delta = self.CBS(t_delta, T, K,
                                St_delta, rt_delta, sigmat_delta)
      self.__init_loss()
      self.__init_linearized_loss()

    def __init_loss(self):
      Vt = self.cbs.black_scholes_call() - self.cbs.rfc_delta()*self.cbs.S
      Vt_delta = self.cbs_delta.black_scholes_call() - self.cbs.rfc_delta()*self.cbs_delta.S
      self.loss = -1 * (Vt_delta - Vt)

    def __init_linearized_loss(self):
      Ldelta = -1*(self.cbs.rfc_theta()*self.delta + self.cbs.rfc_delta()*self.cbs.S*self.x1_delta + self.cbs.rfc_rho()*self.x2_delta + self.cbs.rfc_vega()*self.x3_delta)

      self.Ldelta_mean = -1*self.cbs.rfc_theta()*self.delta
      print("E[Ldelta]:",self.Ldelta_mean)
      self.Ldelta_variance = (self.cbs.rfc_vega()**2)*self.cov_mat[2,2]
      print("V[Ldelta]:",self.Ldelta_variance)

      self.linearized_loss = Ldelta + self.cbs.rfc_delta()*self.cbs.S*self.x1_delta

    def var(self,alpha,method="mc"):
      if method=="lmc":
        return np.quantile(self.linearized_loss, alpha)
      elif method=="mc":
        return np.quantile(self.loss, alpha)
      elif method=="vc":
        return self.Ldelta_mean + np.sqrt(self.Ldelta_variance)*sp.stats.norm.ppf(alpha)

    def var_mean(self,alpha,method="mc"):
      if method=="lmc":
        return np.quantile(self.linearized_loss, alpha) - self.linearized_loss.mean()
      elif method=="mc":
        return np.quantile(self.loss, alpha) - self.loss.mean()
      elif method=="vc":
        return np.sqrt(self.Ldelta_variance)*sp.stats.norm.ppf(alpha)

    def es(self,alpha,method="mc"):
      if method=="lmc":
        return self.linearized_loss[self.linearized_loss >= self.var(alpha,method)].mean()
      elif method=="mc":
        return self.loss[self.loss >= self.var(alpha,method)].mean()
      elif method=="vc":
        return self.Ldelta_mean + np.sqrt(self.Ldelta_variance)*sp.stats.norm.pdf(sp.stats.norm.ppf(alpha))/(1-alpha)

NO_OF_TRADING_DAYS=262

Nsim=10000

t=0
T=0.5
rt=0.05
sigmat=0.2
St=100
K=100
Delta=1/NO_OF_TRADING_DAYS

rfc_stddev_vec=np.array([1e-3,0,1e-4])
rfc_mean_vec=np.array([0,0,0])
corr_mat=np.array([[1,0,-0.5],
                   [0,1,0],
                   [-0.5,0,1]])

dhco = DHCO(0,T,K,St,rt,sigmat,Delta,rfc_mean_vec,rfc_stddev_vec, corr_mat, Nsim)

def print_results(alpha,fun):
  for method in ["mc","lmc","vc"]:
    print(f"method:{method:>3} {fun.__name__+':':10}{fun(alpha,method):.4f}")

for alpha in [0.95,0.99]:
  print(f"{10*'*'}alpha:{alpha}{10*'*'}")
  print_results(alpha,dhco.var)
  print_results(alpha,dhco.var_mean)
  print_results(alpha,dhco.es)