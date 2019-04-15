# Marginalisation & Conditioning

Marginalisation and conditioning are two key components of probabilistic reasoning and are fundamental for computing useful properties about probability distributions.

**Marginalisation**: integrating over a particular variable (or variables) of a multivariate probability distribution. In the two dimensional case we are essentially only considering the values of X regardless of the value of Y. Therefore, for each value of X we need to consider all corresponding values of Y. Formally this is given the the following equation:
$$
p_X(x) = \int_y(p_{X,Y}(x,y)\ dy = \int_y  p_{X \mid Y}(x \mid y)p_Y(y)\ dy
$$
Visually, this can be represented as taking the projection of the 2D space onto the 1D line defined by axis being marginalised for. 

**Conditioning**: given that we know a given value from one variable, what is the probability distribution of the other variable? Visually this can be interpreted as taking an axis aligned slice through the the multivariate probability distribution. The general formula for conditioning is:
$$
p_X(x \mid Y = y) = \frac{p_{X,Y}(x,y)}{p_Y(y)}
$$

