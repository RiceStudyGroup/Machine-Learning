## 1. what is ROC curve?



Receiver operating characteristic (Wikipedia)

https://en.wikipedia.org/wiki/Receiver_operating_characteristic

illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.

<img src="https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2014/11/ROC1.png" width="450px"/>

if there is no overlapping, we can reach a (0, 1) in our ROC curve, i.e., there is a threshold we can use to separate two classes, also means it's linear separable.



https://machinelearningmastery.com/assessing-comparing-classifier-performance-roc-curves-2/

<img src="https://www.medcalc.org/manual/_help/images/roc_intro2.png" width="250px"/>

the closer the ROC curve is to the upper left corner, the higher the overall accuracy of the test

https://www.medcalc.org/manual/roc-curves.php

![](https://ws3.sinaimg.cn/large/006tNc79ly1fzssqn38hwj30ua0u0n05.jpg)

True positive, true means the result is right, and positive means the prediction is positive.

False positive means the result is wrong, and the prediction is positive.

<img src="https://upload.wikimedia.org/wikipedia/commons/3/36/ROC_space-2.png" width="450px"/>

https://youtu.be/OAl6eAyP-yo

## 1.1 what is AUC?

AUC means the area under the ROC curve.

The maximum is 1, and it will be better to be closer to 1.



## 2. what is Precision and recall?


![](https://ws2.sinaimg.cn/large/006tNc79ly1fzssv8p0z6j31200g2mzw.jpg)

![](https://ws4.sinaimg.cn/large/006tNc79ly1fzssubj0n9j30c808y74k.jpg)

precision: the number of your prediction = Positive
recall: the number of the ground truth = Positive

https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c

Precision and recall are two extremely important model evaluation metrics. While precision refers to the percentage of your results which are relevant, recall refers to the percentage of total relevant results correctly classified by your algorithm. `Unfortunately, it is not possible to maximize both these metrics at the same time, as one comes at the cost of another.`

<img src="https://cdn-images-1.medium.com/max/1600/1*DIhRgfwTcxnXJuKr2_cRvA.png" width="250px"/>

So we can train the model based on F1 score. 

## 3. what is accuracy ?

![](https://ws4.sinaimg.cn/large/006tNc79ly1fzst1vv9gnj30o204cwf0.jpg)

---
