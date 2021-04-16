---
layout: post
title:  "Neural Networks with Aleatoric Uncertainty"
date:   2021-04-04 19:31:18 +0100
categories: math probability torch
---

In regression problems, neural networks can be used to estimate the degree of uncertainty in their outputs. Broadly speaking, [uncertainty can be split into two types](https://arxiv.org/pdf/1910.09457.pdf):

 - Aleatoric Uncertainty: uncertainty arising due to the data generating process. Aleatoric uncertainty cannot be reduced by collecting more data.
 - Epistemic Uncertainty: uncertainty arising due to limitations of the model itself. In principle this can be reduced by using a better model or collecting more data. 

Here we model show a brief example of modelling aleatoric uncertainty using `torch.distributions` to predict both the mean value of a regression problem, but also the variance over a set of data points. 

As a toy problem, we will fit the following sine function with [heteroscedastic](https://en.wikipedia.org/wiki/Heteroscedasticity) noise. We'll assume that each data point is Gaussian distributed with both unknown mean and unknown variance.

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.distributions as td
```

```python
def true_function(x):
  y = np.sin(x*10) + 1
  return y

def noisy_function(x):
  y = true_function(x)
  noise = stats.norm(loc=0, scale=x*.5).rvs(x.shape[0])
  return y + noise

x = np.linspace(0, 1, 1000)
y = noisy_function(x)

plt.scatter(x, noisy_function(x), label='Noisy Samples', alpha=0.5, edgecolors='black')
plt.plot(x, true_function(x), c='r', label='True Function', linewidth=3)
plt.legend()
```
    
{:refdef: style="text-align: center;"}
![png](/assets/media/nn_with_uncertainty_files/nn_with_uncertainty_2_1.png)
{: refdef}


Below we create a simple feedforward network with a two layers. We use leaky ReLU for the activation function. Notice that we split the last layer in order to parameterise a Gaussian distribution. The variance term is passed through the exponential function to ensure the value is constrained to always be positive.

We can then fit the function by minimising the log likelihood of the data under the model.


```python
class Feedforward(torch.nn.Module):
  def __init__(self, input_size, hidden_size):
      super(Feedforward, self).__init__()
      self.fc1 = torch.nn.Linear(input_size, hidden_size)
      self.fc2 = torch.nn.Linear(hidden_size, 2)

  def forward(self, x):
      hidden = F.leaky_relu(self.fc1(x))
      hidden = F.leaky_relu(self.fc2(hidden))
      return td.Normal(hidden[:, :1], torch.exp(hidden[:, 1:]))

np.random.seed(0)
torch.manual_seed(0)

#Generate synthetic data
x_train = np.linspace(0, 1, 1000)
y_train = noisy_function(x_train)

# setup data loaders
X = torch.tensor(x_train).float()
Y = torch.tensor(y_train).float()
data_loader = DataLoader(TensorDataset(X.unsqueeze(1), Y.unsqueeze(1)), batch_size=100,
                              pin_memory=True, shuffle=True)

# create network
net = Feedforward(1, 50)
net.train()

optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)

# train model
epoch = 1000
for epoch in range(epoch):
  for x, y in data_loader: 
    optimizer.zero_grad()
    # Forward pass
    dist = net(x)

    # compute negative log likelihood
    loss = -dist.log_prob(y).mean()
   
    # Backward pass
    loss.backward()
    optimizer.step()
  
  if epoch % 100 == 0:
    print(f'Epoch {epoch:4}, nll: {loss.item():.4f}')
```

    Epoch    0, nll: 1.2149
    Epoch  100, nll: 0.5191
    Epoch  200, nll: 0.1218
    Epoch  300, nll: -0.1060
    Epoch  400, nll: 0.0076
    Epoch  500, nll: -0.1632
    Epoch  600, nll: -0.0061
    Epoch  700, nll: -0.2909
    Epoch  800, nll: -0.1450
    Epoch  900, nll: -0.0618


Finally, we can plot the resulting fitted function along with the estimated error bars for our toy function. We can see that both the mean and variance of the estimated points follow our true function fairly well.  


```python
net.eval()
with torch.no_grad():
  X_hat = torch.tensor(np.linspace(0, 1, 30).reshape(-1, 1)).float()
  dist = net(X_hat)

mus, sigmas = dist.loc.numpy().squeeze(), dist.scale.numpy().squeeze()
plt.errorbar(X_hat.squeeze(), mus, sigmas, label='Model Function',
             linewidth=3, color='lime', fmt='o-',capsize=5, elinewidth=2)
plt.plot(X_hat.squeeze(), true_function(X_hat.squeeze()), label='True Function',
         linewidth=3, color='r')
plt.scatter(x_train, y_train, label='Noisy Samples', alpha=0.5, edgecolors='black')
plt.legend()
```

    
{:refdef: style="text-align: center;"}
![png](/assets/media/nn_with_uncertainty_files/nn_with_uncertainty_6_1.png)
{: refdef}
