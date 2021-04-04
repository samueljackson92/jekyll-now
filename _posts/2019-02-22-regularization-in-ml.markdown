---
layout: post
title: L1 and L2 regularization for Machine Learning
date: 2019-02-22
tags: machine-learning regularization math Bayesian
---

Regularisation in machine learning is an important concept to combat overfitting. Overfitting is the failure of a model to generalise well to unseen data. Essentially, overfitting occurs when the model ceases to fit to the underlying objective function of the data and begins to fit to noise.

![img](https://lh6.googleusercontent.com/NiJa3ahGchd7a8Bzn77EKSGCZV35ay-4spfH3-E6sFN-R1CO81uE_ct0CU7vITAbkwZpq4JKWRcx85iunXpwmEiAMkBGZqL51PcaIFDfw0RV-YizPvil-EleijCJLpIaSUX5zlpeKq4)

Let's say that we're working with a simple linear model and wish to find the best fit to the data:


$$
\min_\theta \sum\limits_{i=1}^N V(x_i \cdot \theta, y_i)
$$


Where:

- $x_i$ are the features of the model.
- $y_i$ are the labels for a particular training instance.
- $\theta$ are the parameters/weights of the model which we wish to learn.
- $V$ is the loss function which we wish to minimise.

L1 and L2 regularisation are two common regularization methods applied in machine learning. Both aim to prevent overfitting by penalising the weights of a model in some way. In a very general form they both add an additional term to the equation above that is only dependant on the parameters $\theta$


$$
\min_\theta \sum\limits_{i=1}^N V(x_i \cdot \theta, y_i) + \lambda R(\theta)
$$


Where:

- $R$ is the regularisation scheme (i.e. L1 or L2).
- $\lambda$ is an adjustable parameter controlling how much influence the regularisation term has. Typically this is set to a small value and chosen based on experimentation.

## L2 Regularisation

L2 regularisation is defined as follows:


$$
R(\theta) = \left\lVert \theta \right\rVert _2 = (\sum\limits_{i=0}^N \theta_i^2)^\frac{1}{2}
$$


While this might look complicated this is really just the Euclidean norm of the vector of parameters $\theta$. It can easily be computed by taking the dot product of the parameters $\theta$ with itself $\sqrt{\theta^T\theta}$. Note that it is more computationally efficeint to just work with the squared norm to avoid having to compute the square root.



## L1 Regularisation

L1 regularisation is defined as follows:


$$
R(\theta) = \left\lVert \theta \right\rVert _1 = \sum\limits_{i=0}^N |\theta_i|
$$


That is, it is simple the sum of the absolute value of the components of $\theta$. An important property of L2 regularisation is that it encourages sparsity in the solution. In other words, it disriminates well between elements which are zero or almost zero and elements which far from zero. This can be useful as a built in form of feature selection. Features with weights which are almost zero may be dropped to simplify the model. 

It is worth noting that L1 regularisation is more difficult to work with than the L2 regularisation. It is not a differentiable function so can be difficult to work with in an optimisation context.



## LP Norms

L1 & L2 regularisation both belong the the family of $L_p$-norms. The 1 & 2 in their name comes from the fact that they are the 1st and 2nd p-norm. The general p-norm equation is


$$
L_p(\theta) = \left\lVert \theta \right\rVert _p = (\sum\limits_{i=0}^N \theta_i^p)^\frac{1}{p}
$$


$L$ norms for different values of $p$ are plotted in the figure below. It can be seen that the $L_1$ norm is much more "spikey" than the $L_2$ norm. In the context of regularisation, the minimum of the regularisation term will always occur at one of the corners of the $L_1$ ball which implies that one coordinate will always be ~0. On the other hand, the $L_2$ norm is the Euclidean ball, which does not have a sharp intersection along one axis. The minimum will therefore occur at an intersection with the ball which is not necessarily ~0 along an axis. 



![img](https://cdn-images-1.medium.com/max/1440/1*OLFN24vF_c3y5p3tiz4_5A.png)

## A Bayesian Viewpoint

In a Bayesian context the regularization term in the first equation above can be reformulated as a prior on the parameters of the model. In a Bayesain viewpoint the problem of inferring the model parameters $\theta$ given that we have known $x$ and $y$. In other words we must solve:



$$
P(\theta \mid x, y) = \frac {P(y \mid x, \theta) \cdot P(\theta)}{\int P(y \mid x,\theta) d\theta}
$$


Where 

- $P(y\mid x,\theta)$ is the likelihood of observing data $y$ given features $x$ and parameters $\theta$
- $P(\theta)​$ is the our prior belief about the parameters $\theta​$

For a simple linear model, and using least squares as a loss function. We could assume for example that that


$$
P(y \mid x,\theta) \sim N(x^T\theta, \sigma^2)
$$


When we minimise the cost function $V$ without any regularising term we are essentially find the maximum likelihood estimate of $P(y \mid x,\theta)$. By adding a regularising term to the model which is only dependant on the parameters $\theta$ we are essentially adding a prior belief in what values of  $\theta$  are more "reasonable" (smaller weights for $L_2$ and sparser weights for of $L_1$). Or to put it another way, we're stating an assumption about how we think the parameters $\theta$ ought to be distrubted.

In the case of $L_2$ regualrisation we state that the prior distribution is normally distributed. i.e.


$$
P(\theta) \sim N(0, \sigma^2)
$$


Which can be shown to be equivilent to applying adding a $L_2$ regularisation term. In the case of $L_1$ regularisation we state that the prior distribution follows a Laplacian distribution. i.e.


$$
P(\theta) \sim Laplace(0, b)
$$


Which can be shown to be equivilent to applying adding a $L_1$ regularisation term. This can be seen by viewing the two functions $R(\theta)$ as the maximum likelihood estimators for a Gaussian/Laplace distribution centered on zero.

## Further Reading

- Bishop, Christopher M. *Pattern recognition and machine learning*. springer, 2006.
- [Bayesian Interpretations of Regularization](http://www.mit.edu/~9.520/spring09/Classes/class15-bayes.pdf)

