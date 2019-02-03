## Derive the normal equation for linear regression.

## Prove that for linear regression MSE can be derived from maximal likelihood by proper assumptions.

what is MSE?

https://en.wikipedia.org/wiki/Mean_squared_error

![](https://ws1.sinaimg.cn/large/006tNc79ly1fzsps8cll2j30cg03ydfy.jpg)

## Explain to me what is Locally weighted linear regression

## does the l1 l2 weight decay including the bias? or just weights.

No,

cause the bias is like the mean of our prediction, and it make no sense to make it close to 0 as possible. But we can let weights to be close to 0 to make the model simple.

## 1. Explain to me what is LR?


## 2. what is the loss function of LR?


Basically it's the mean of square of difference between prediction and ground truth.

we can explain this loss function by assume the error is following a Gaussian distribution, with mean \mu equals to 0 and a variance.

The variance actually will not influence the model.


## 3. how do we get the best parameters?

we can use gradient descent to approach the best parameters and bias?

Or we can use normal equation.
But to calculate the inverse of matrix, it's very computational expensive.

What about Newton method? We need to calculate the Hessian matrix. So

https://www.quora.com/Does-it-make-sense-to-use-the-Newton-method-for-linear-regression-Does-it-make-sense-to-use-the-curvature-information-in-this-context

![](https://ws2.sinaimg.cn/large/006tNc79ly1fzspmccx3ij30x009kq5s.jpg)

## 4. Is linear regression a convex problem? i.e. there is only one local minimum?

Yes, it's a convex optimization problem.

https://www.quora.com/Why-is-linear-regression-a-convex-optimisation-problem

## 5. do we need to do hyper parameters tune for linear regression?

if you use regularization, you can change the hyper parameter that controls the power of regularization.
or you can change the learning rate.

## 6. what is the difference between gradient descent and batch and stochastic?

For large data set, stochastic is faster? From Andrew's notes.




# Linear Regression


The model is to train some parameters and bias to predict a float value.

the value we predict is different with the real target.

The cost function is least squares.

It's an optimization problem, to minimize the cost function.




## 1. LMS algorithm.

`What does LMS mean ?`

LMS means least mean squares.

`why gradient descent work?`


## 2. Normal equations

as long as the
