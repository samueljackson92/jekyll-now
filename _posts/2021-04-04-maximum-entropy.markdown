---
layout: post
title:  "Maximum Entropy & Jaynes' Dice"
date:   2021-04-04 19:31:18 +0100
categories: math Bayesian probability
---


[Maximum entropy](https://en.wikipedia.org/wiki/Principle_of_maximum_entropy) is method to find maximally uninformative probability distribution given some known information about the distribution. The principle of maximum entropy as a number of uses, such as providing a formal way to pick an appropriate prior distribution while making the fewest possible assumptions about the data.

## Jaynes' Dice

A very simple example of where maximum entropy can be useful is the following problem, often referred to as Jaynes' Dice after the Physicist [E.T. Jaynes](https://en.wikipedia.org/wiki/Edwin_Thompson_Jaynes).

> *Assume you are told the average value of a biased dice roll is 4.5. How should we assign probability such that the dice are maximally uninformative way?*

For a fair dice, we would intuitively assign $p(x_i) = \frac{1}{6}$ for each face, yield a uniform distribution as a prior. This has an expected value of $3.5$. But what about the case where the dice is biased, but we only know the expected value and nothing else?

The expected value in this problem is an example of *testable information*. Testable information is just a statement about the probability distribution which is taken as a fact. Given we know some testable information, we'd like to find the probability distribution that makes the least assumptions about the data. This is what the principle of maximum entropy can provide for us.

## Entropy

[Information entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) is a measure of the average level of uncertainty in a variable's outcome. Entropy is high when there is a lot of uncertainty, and low when when outcome is very deterministic. For example, a fair coin with $p(H) = \frac{1}{2}$ has very high (in fact maximum) entropy. While a two headed coin with $p(H) = 1$ has entropy of zero: the outcome is entirely deterministic.  

For discrete distributions, information entropy is defined with the following equation:

$$
H(X) = -\sum_i^{N} p(x_i)\ \log_2 p(x_i)
$$

The choice of logarithm function is slightly arbitrary and only changes the units of the resulting function. Using the $\log_2$ function results in entropy being measured in units of bits. Another common choice is the natural logarithm ($\ln$), resulting in units of nats. 

Returning to our dice example, we can calculate the entropy for different distributions of biased dice below:


```python
from scipy import optimize
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
np.random.seed(42)
```

```python
def entropy(p_x):
  return -(p_x*np.log2(p_x, where=p_x > 0)).sum()

x = np.arange(1, 7)

p_x_fair = np.array([1/6]*6)
fair_dice_entropy = entropy(p_x_fair)

p_x_unfair = np.array([.9, .01, .01, .01, .01, .01])
unfair_dice_entropy = entropy(p_x_unfair)

p_x_det = np.array([1, 0, 0, 0, 0, 0])
deterministic_dice_entropy = entropy(p_x_det)

# Plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))
ax1.bar(x, p_x_fair)
ax1.set_title(f'Fair dice\n Entropy:\n{fair_dice_entropy:.3f} bits')
ax2.bar(x, p_x_unfair)
ax2.set_title(f'Unfair dice\n Entropy:\n{unfair_dice_entropy:.3f} bits')
ax3.bar(x, p_x_det)
ax3.set_title(f'Deterministic dice\n Entropy:\n{deterministic_dice_entropy:.3f} bits')

for ax in fig.axes:
  ax.set_ylabel('$p(x)$')
  ax.set_xlabel('$x$')

plt.tight_layout()
```


    
![png](/assets/media/maximum_entropy_files/maximum_entropy_2_0.png)
    


We can see that the lower the entropy, the more deterministic the outcome of the dice roll will be. The higher the entropy, the more uninformative the distribution is.

### Maximum Entropy
Now we can begin to see how to solve the problem of Jaynes' dice. We wish to find the assignment of probabilities to dice faces that maximise the entropy given some constraint, such as the mean of the distribution.

This is essentially a optimisation problem with constraints. We can use the method of Lagrange multipliers to formulate a optimisation problem with our additional equality constraints. 

Below we actually get to the full case incorporating our testable information, we can solve a slightly simpler problem: finding the maximum entropy distribution only the constraint that the probability assignments must such to $1$. That is:

$$
-\sum_i^{N} p(x_i)\ \log_2 p(x_i) - \lambda (1 - \sum_i^N p(x))
$$

The first part is just the entropy equation we wish to maximise. The second part is the normalisation constraint to ensure the probability distribution sums to $1$. $\lambda$ is the Lagrange multiplier. We can solve this optimisation problem in Python with just a few lines of code:


```python
def func(p_x):
  """ Negative Entropy function 
  
  Negative here because we want to use scip minimize to 
  perform the optimization.
  """
  return (p_x*np.log2(p_x)).sum()

def con_norm(p_x):
  """ Normalization condition """
  return 1 - p_x.sum()

# Our 6 values for each dice face.
x = np.arange(1, 7)

# Intialise probability distribution to a random guess.
p_x = np.random.uniform(size=len(x))
p_x /= p_x.sum()

# minimise the function
out = optimize.minimize(func, p_x, constraints=[{'type': 'eq', 'fun': con_norm}])

plt.bar(x, out['x'])
plt.title(f'Entropy: {entropy(out["x"]):.3f}, Expectation: {np.average(x, weights=out["x"]):.3f}')
plt.ylabel('$p(x)$')
plt.xlabel('$x$')
```

![png](/assets/media/maximum_entropy_files/maximum_entropy_4_2.png)
    


We see that optimising the entropy with only the normalisation constraint yield the uniform distribution. This is exactly what we'd expect to assign as a sensible prior given that we know nothing else about the distribution. 

Finally, what about the case where we do know something about the distribution, such as the mean? The can simply add that as a constraint to the maximum entropy equation above and again solve using the method of Lagrange multipliers:

$$
-\sum_i^{N} p(x_i)\ \log_2 p(x_i) - \lambda_1 (1 - \sum_i^N p(x)) - \lambda_2 (\mu - \mathbb{E}[X]) 
$$

The final additional term is our optimisation constraint for the testable information. $\mu$ is our known mean value ($4.5$ for example) and $\mathbb{E}[X]$ is the expected value of our categorical probability distribution. Optimizing as before with the additional constraint we get the following:



```python
# Testable information: the mean of the distribution is mu
mu = 4.5

def con_mu(p_x):
  """ Mean constraint """
  return mu - np.average(x, weights=p_x)

out = optimize.minimize(func, p_x, constraints=[{'type': 'eq', 'fun': con_norm}, {'type': 'eq', 'fun': con_mu}])

plt.bar(x, out['x'])
plt.title(f'Entropy: {entropy(out["x"]):.3f}, Expectation: {np.average(x, weights=out["x"]):.3f}')
plt.ylabel('$p(x)$')
plt.xlabel('$x$')
```
    
![png](/assets/media/maximum_entropy_files/maximum_entropy_6_2.png)
    


Which assigns higher weights to dice faces with higher numbers. This intuitively makes sense given that our expected value is higher than that of a fair dice. This must mean we are more likely to roll higher valued faces more than lower valued faces!

### Further Reading

 - [Maximum Entropy Distributions - Brian Keng](https://bjlkeng.github.io/posts/maximum-entropy-distributions/)
 - [Concentration of distributions at entropy maxima - E. T. Jaynes](https://bayes.wustl.edu/etj/articles/entropy.concentration.pdf)
 - Sivia, Devinderjit, and John Skilling. Chapter 5. Data Analysis: a Bayesian Tutorial. OUP Oxford, 2006.
