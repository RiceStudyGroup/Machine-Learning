## 1. explain to me what is logistic regression?

similar to linear regression, but we will map the result to a value between 0 and 1 to represent the probability that this sample belong to the target class or not.


## 2. what is the sigmoid function?


![](https://ws1.sinaimg.cn/large/006tNc79ly1fzsq5bu4rbj309y04omx6.jpg)

what's its derivative?


![](https://ws1.sinaimg.cn/large/006tNc79ly1fzsq5zh5nrj30ns0dqaax.jpg)

## 3. what is the cost function?

Since those p represent the probabilities. We can use the MLE to get the cost function.


![](https://ws4.sinaimg.cn/large/006tNc79ly1fzsq70jwsfj30mc02mmxe.jpg)


![](https://ws4.sinaimg.cn/large/006tNc79ly1fzsqqv2m82j30x00m6mzj.jpg)

Then -l(theta) is the cost function and we want to minimize it, i.e., to maximize the likelihood l(theta).


## 4. what should we do if there're more than 2 classes?
