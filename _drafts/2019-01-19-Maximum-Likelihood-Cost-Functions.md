---
layout: post
title: Maximum Likelihood Cost Functions
---

[Maximum Likelihood estimation][MLE] (MLE) is the process of estimating the parameters for a statistical model by finding the maximum of the likelihood function. As machine learning algorithms are really just statistical models MLE, can be used to define some measure of what the "best" parameters are for our model. In other words, it can be used to derive a cost function used to train a model.

### MLE for Linear Regression
Let's take one of the simplest machine learning algorithms and show that deriving the MLE produces a sensible cost function. For this example we'll use plain old linear regression. For the case of a single parameter linear regression is defined as follows:

$$
\begin{align*}
    & y = x\beta + \epsilon
\end{align*}
$$ 

### References

1. [Probability concepts explained: Maximum likelihood estimation](https://towardsdatascience.com/probability-concepts-explained-maximum-likelihood-estimation-c7b4342fdbb1)

[MLE]: https://en.wikipedia.org/wiki/Maximum_likelihood_estimation
