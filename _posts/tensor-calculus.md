---
layout: post
title: Tensor Calculus
date: 2000-01-01
tags: machine-learning, math, tensors
---

Tensor calculus is the extension of vector calculus to tensor fields. The rules for manipulating tensors and tensor fields are described by the Ricci calculus, named after Gregorio Ricci-Curbastro who first published the system in 1900. The primary motivating idea is that the solution at the end of the calculation is coordinate system independant. The foundation of Ricci calculus is to split the vector world into two parts: vectors and covectors.

**Covector**: A linear map that transforms a vector from a vector space to a real number. I.e. $f : V \rightarrow \mathbb{R}​$ and the following conditions hold:
$$
f(\mathbf{x} + \mathbf{y}) = f(\mathbf{x}) + f(\mathbf{y}) \\
f(a\mathbf{x}) = af(\mathbf{x})
$$
A linear functional is also called a *line form*. One example of a linear functional is the dot product of two vectors. The row vector effectively contains the coefficients of the functional.

Ricci calculus proposes a notation for making the difference between the two forms explicit. Vectors are represented by up indices $x^i$. Covectors are represented by down indices $y_i$. In this notation, it is never valid to take the dot product between two vectors or two covectors. Only the product between a covector and a vector is allowed. More explictily, the following is allowed:
$$
\sum\limits_{i=1}^{N} x_i y^i
$$
But not
$$
\sum\limits_{i=1}^{N} x^i y^i \\
\sum\limits_{i=1}^{N} x_i y_i
$$
A tensor contraction is when the same index appears twice, once as a lower index and once as an upper index, in a summation. This is such a common operation that the summation notation is dropped and the Einstien summation convention is used:
$$
\sum\limits_{i=1}^{N} x_i y^i \equiv x_iy^i
$$




### References

- S¨oren Laue, Matthias Mitterreiter, and Joachim Giesen. Computing higher
  order derivatives of matrix and tensor expressions. In Advances in Neural
  Information Processing Systems,

- http://mathworld.wolfram.com/LinearFunctional.html

- https://www.quora.com/What-is-an-intuitive-explanation-of-Ricci-calculus
- https://en.wikipedia.org/wiki/Linear_form