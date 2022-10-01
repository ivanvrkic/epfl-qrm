import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats

Nsim=10000

St=100
rt=0.05
sigmat=0.2

T=1
K=100
Delta=1/252

rfc_stddev_vec=np.array([1e-2, 1e-4, 1e-3])
rfc_mean_vec=[0,0,0]
corr_mat=np.array([[1,0,-0.5],
          [0,1,0],
          [-0.5,0,1]])

cov_mat = np.matmul(rfc_stddev_vec.reshape(-1,1),rfc_stddev_vec.reshape(1,-1)) * corr_mat

x1_delta,x2_delta,x3_delta = np.transpose(sp.stats.multivariate_normal.rvs(mean=rfc_mean_vec, cov=cov_mat, size=Nsim))

class ECO:
  class CBS:
    def __init__(self,t,T,K,S,r,sigma):
      self.t=t
      self.T=T
      self.K=K
      self.S=S
      self.r=r
      self.sigma=sigma

      self.d1 = (np.log(self.S/self.K) + (self.r+(self.sigma**2)/2)*(self.T - self.t))/(self.sigma*np.sqrt(self.T - self.t))
      self.d2 = self.d1 - self.sigma*np.sqrt(self.T - self.t)

    def black_scholes(self):
      return self.S*sp.stats.norm.cdf(self.d1) - np.exp(-self.r*(self.T - self.t))*self.K*sp.stats.norm.cdf(self.d2)
    def rfc_theta(self):
      theta = -self.S*sp.stats.norm.cdf(self.d1)*self.sigma/(2*np.sqrt(self.T-self.t)+self.r*np.exp(-self.r*(self.T-self.t))*K*sp.stats.norm.cdf(self.d2))
      return theta
    def rfc_delta(self):
      delta = sp.stats.norm.cdf(self.d1)
      print("Delta:",delta)
      return delta
    def rfc_rho(self):
      rho = (self.T-self.t)*np.exp(-self.r*(self.T-self.t))*self.K*sp.stats.norm.cdf(self.d2)
      print("Rho:",rho)
      return rho
    def rfc_vega(self):
      vega = self.S*sp.stats.norm.cdf(self.d1)*np.sqrt(self.T-self.t)
      print("Vega:",vega)
      return vega

  def __init__(self,t,T,K,S,r,sigma,delta,x1_delta,x2_delta,x3_delta):
    self.cbs = self.CBS(0,T,K,S,r,sigma)

    self.delta = delta
    self.x1_delta = x1_delta
    self.x2_delta = x2_delta
    self.x3_delta = x3_delta

    St_delta = S*np.exp(x1_delta)
    rt_delta=abs(r+x2_delta)
    sigmat_delta=abs(sigma+x3_delta)
    t_delta=t+delta

    self.cbs_delta = self.CBS(t_delta,T,K,St_delta,rt_delta,sigmat_delta)

  def loss(self):
    return  -1*(self.cbs_delta.black_scholes()-self.cbs.black_scholes())
  def linearized_loss(self):
    return -1*(self.cbs.rfc_theta()*self.delta + self.cbs.rfc_delta()*self.cbs.S*self.x1_delta + self.cbs.rfc_rho()*self.x2_delta + self.cbs.rfc_vega()*self.x3_delta)

def plot_loss(x,plot_title,save_filename=None):
  x_range = np.linspace(x.min(), x.max(), Nsim)
  pdf = sp.stats.norm.pdf(x_range,x.mean(),x.std())
  plt.hist(x, bins=100, density=True, linewidth=0.3,label='Empirical distribution')
  plt.plot(x_range, pdf, linewidth=2.5,label=f'Normal PDF')
  plt.title(plot_title)
  plt.xlabel('Simulated Loss')
  plt.ylabel('Density')
  plt.legend()
  if save_filename:
    plt.savefig(save_filename)

calloption = ECO(0,T,K,St,rt,sigmat,Delta,x1_delta,x2_delta,x3_delta)

Lt_t_delta = calloption.loss()
plot_loss(Lt_t_delta,'Simulated Loss and Normal PDF',save_filename="SimulatedLoss.png")

Ldelta_t_t_delta = calloption.linearized_loss()
plot_loss(Ldelta_t_t_delta,'Simulated Linearized Loss and Normal PDF',save_filename="SimulatedLinearizedLoss.pdf")