## 1. https://www.mitbbs.com/article_t/JobHunting/33120253.html

Machine learning related questions:
-  Discuss how to predict the price of a hotel given data from previous
years
-  SVM formulation
-  Logistic regression
-  Regularization
-  Cost function of neural network
-  What is the difference between a generative and discriminative algorithm
-  Relationship between kernel trick and dimension augmentation
-  What is PCA projection and why it can be solved by SVD  
-  Bag of Words (BoW) feature
-  Nonlinear dimension reduction (Isomap, LLE)
-  Supervised methods for dimension reduction
-  What is naive Bayes
-  Stochastic gradient / gradient descent
-  How to predict the age of a person given everyone’s phone call history
-  Variance and Bias (a very popular question, watch Andrew’s class)
-  Practices: When to collect more data / use more features / etc. (watch
Andrew’s class)
-  How to extract features of shoes
-  During linear regression, when using each attribute (dimension)
independently to predict the target value, you get a positive weight for
each attribute. However, when you combine all attributes to predict, you get
some large negative weights, why? How to solve it?
-  Cross Validation
-  Reservoir sampling
-  Explain the difference among decision tree, bagging and random forest
-  What is collaborative filtering
-  How to compute the average of a data stream (very easy, different from
moving average)
-  Given a coin, how to pick 1 person from 3 persons with equal probability.


Coding related questions:
-  Leetcode: Number of Islands
-  Given the start time and end time of each meeting, compute the smallest
number of rooms to host these meetings. In other words, try to stuff as many
meetings in the same room as possible
-  Given an array of integers, compute the first two maximum products(乘积)
of any 3 elements (O(nlogn))
-  LeetCode: Reverse words in a sentence (follow up: do it in-place)
-  LeetCode: Word Pattern
-  Evaluate a formula represented as a string, e.g., “3 + (2 * (4 - 1) )”
-  Flip a binary tree
-  What is the underlying data structure for JAVA hashmap? Answer: BST, so
that the keys are sorted.
-  Find the lowest common parent in a binary tree
-  Given a huge file, each line of which is a person’s name. Sort the names
using a single computer with small memory but large disk space
-  Design a data structure to quickly compute the row sum and column sum of
a sparse matrix  
-  Design a wrapper class for a pointer to make sure this pointer will
always be deleted even if an exception occurs in the middle
-  My Google onsite questions: http://www.mitbbs.com/article_t/JobHunting/33106617.html

面试的一点点心得：
最重要的一点，我觉得是心态。当你找了几个月还没有offer，并且看到别人一直在版
上报offer的时候，肯定很焦虑甚至绝望。我自己也是，那些报offer的帖子，对我来说
都是负能量，绝对不去点开看。这时候，告诉自己四个字：继续坚持。我相信机会总会
眷顾那些努力坚持的人，付出总有回报。
machine learning的职位还是很多的，数学好的国人们优势明显，大可一试, 看到一些
帖子说这些职位主要招PhD，这个结论可能有一定正确性。但是凭借我所遇到的大部分
面试题来看，个人认为MS或者PhD都可以。MS的话最好有一些学校里做project的经验。
仔细学习Andrew Ng在Coursera上的 machine learning课，里面涵盖很多面试中的概念
和题目。虽然讲得比较浅显，但对面试帮助很大。可以把video的速度调成1.5倍，节省
时间。
如果对一些概念或算法不清楚或者想加深理解，找其他的各种课件和视频学习，例如
coursera，wiki，牛校的machine learning课件。
找工作之前做好对自己的定位。要弄清楚自己想做什么，擅长做什么，如何让自己有竞
争力，然后取长补短（而不是扬长避短）。
感觉data scientist对coding的要求没有software engineer那么变态。不过即便如此
，对coding的复习也不应该松懈。


我个人觉得面试machine learning相关职位前需要熟悉的四大块：
Classification:
Logistic regression
Neural Net (classification/regression)
SVM
Decision tree
Random forest
Bayesian network
Nearest neighbor classification

Regression:
Neural Net regression
Linear regression
Ridge regression (add a regularizer)
Lasso regression
Support Vector Regression
Random forest regression
Partial Least Squares

Clustering:
K-means
EM
Mean-shift
Spectral clustering
Hierarchical clustering

Dimension Reduction:
PCA
ICA
CCA
LDA
Isomap
LLE
Neural Network hidden layer


---

## 2. https://www.1point3acres.com/bbs/thread-236437-1-1.html



1. 国人女，ML和data pipeline背景都有吧，问了一下做的一个big data pipeline project，之后是一个machine learning pipeline，问了一下kmeans的细节和优化等等
之后coding题目，给一个gird，有x和y坐标，每个坐标只能往右走或者往上走，从(0, 0) 开始，走到终点(n, n)，问总共有多少种path，可以用recursion解

2. 国人男，背景更偏ML，问了同样的两个pipeline project，特别的问了一下Kafka细节
之后coding: given a sentence with multiple words, reverse the sentence by words:
example: "hello world", output: "world hello"，解法挺多的吧，deque等等
另一个问题，读一个很大的文件，对每行进行处理，然后存到磁盘，但是文件不能fit into memory，只限单机，如何解决？
讨论了一下，最后他给的解法是一行一行的读然后一行一行的写（什么鬼。。。）

3. 印度男，上来先coding：input a string and a set of substring ("ab", "cd") -> output a new string with all substring removed, e.g. "ccabd" -> "ccd" -> "c"
有点类似于另一道面经题consecutive string，我用的stack解，不过也可以直接暴力解
之后问了Spark streaming内部原理，写了一个简单的map -> groupby流程，但他的问题很tricky
. From 1point 3acres bbs
4. 中东manager，走到了另一栋apple的大楼吃午饭，一直在聊天，都是high level的概念，比如不同database tradeoff，open source的东西，心累，吃饭都吃不好

5. 国人小哥，上来coding，Jump Game的变体，要求输出最小的jump step（每次从一个index到另一个index是一个jump），这题我也没准备，提醒了一下recursive做了出来，小哥挺满意
后来问了同一个machine learning pipeline（看来他们的确很看重ML经验），然后又问了另一个distributed database systems project，问available和consistent的一些问题
然后就是我问他了，其实我感觉他比我紧张

6. 印度男，之前搜他的linkedin好牛的样子，好像是team lead
上来问了现在的工作内容，然后直接问了原题Longest Valid Parentheses，我到这个时候口干舌燥脑子早就不转了，还是没遇到的hard题目，在他的不断提醒下才勉强解出来了，感觉不是很满意
之后又问了machine learning pipeline，PCA kmeans原理

7. 印度男，疑似hiring manager，上来就问在现在的公司干什么，然后就是头脑风暴般的想到哪里问到哪里，都是knowledge base的，而且脸色也很难看，感觉就是想要看看我是否抗压吧：
why leaving your current company, how to build a real-time pipeline, what if the real time data must not be lost (push v.s. pull based message Q), what is storm, Flume, Kafka Spark Streaming, what is the benefit of Spark SQL, fault tolerance with database commit log/checkpoint, what is your best programming language, why do you want to work in a fast growth team, etc..

---

## 3. https://www.1point3acres.com/bbs/thread-457683-1-1.html

Part 1
Tom:请设计一个给user推荐问题的ML 系统
我:在high level有若干approach，1）基于已知question内容，推荐和用户兴趣相关 2）不关心内容，历史点击数据，用每个问题被点击的用户来做collaborative filter 3）一些Topic model的方法，按照topic tag分类
Tom：先从简单的开始，假设所有用户看到的都是同样的推荐问题
我：那么此时选用第一种approach，需要3个部分1）某一种embedding method去吧每个问题从document变成vector，典型的方法有bow 2）选择某种metrc 去计算特定个vector之间的距离，典型的有euclidean dist， cosine similarity 3）选择某种方法筛选出最相近的K个问题推荐给用户，最简单的是挨个计算distance，然后排序返还最相近的k个
Tom：你提到了两种不同的distance，那么你更prefer eucliden还是cosine similarity
我：（此时有点被问住，还是比较水没反应过来，所以生想了一个解释）prefer cosin similarity，因为两者如果使用过同样的词，similarity相近比较make sense，因为是bow embedding 高维空间上小的Euclidean dist并不意味这意思相近
Tom：这个算法complexity如何
我：假设已经有了embedding，每个similarity计算是点乘所以很快，要对每个vector挨个计算此时为O（N），计算后要进行排序所以有平均NLOG(N)。（不知道对不对纯属胡诌，大神请指正）
Tom：这个complext会不会有什么问题
我：（强行往好了说）当然不是最好的，但是nlogN应该属于可以接受的（真不要脸）
Tom：能否进行一些优化
我：可以分布式计算，例如10台计算机，每个分别找出k个最相近的，然后master node使用min heap 分路归并可以快速从10*K个最小的结果中找出最小的k个（不知道是不是一个合理的答案）
Tom：（简要重复了我的之前所有回答，确保他没有错误理解我的答案）

Part 2
Tom：那么这是你说的最简单的方法，如果想要优化一下模型，你能怎么improve
我：使用某种embedding和similarity，然后排序k个最相近的这个approach比较直观，简单，逻辑上也合理。所以这个基本的method可以保留，我会选择想办法优化更好的embeding和metric
Tom：请给出具体的例子-baidu 1point3acres
我：用word2Vec embedding去生成词的vector，因为word2Vec可以把语义融入向量中，（similarity measure开了个脑洞，隐约记得以前看了篇paper），构建一个matrix m，m(i,j)代表第一个问题ith词转换到第二个问题jth词的cost，这个cost使用两个对应词word2vec的Euclidean distance代表，然后对这些所有的cost做加权平均。
Tom：（因为这部分我叙述的setting和方法有点乱，所以Tom反复确认，最后我们都相互confirm理解了对方的意思）
Tom：假设有了两个不同的model，怎样评价那个推荐的更好
我：（我好久没有系统复习Data Science，时候觉得可能需要答一些A/B testing， ROC什么的）可以用用户点击推荐内容的比例来比较，更细致一点的办法是统计用户在每个问题上花费的时间，然后根据问题答案的长度normalize，把这个时间作为score

Part 3
Tom：在最初提到了除了基于context还有别的approach，能否简单说一些如果不基于内容，你打算怎么做
我：构建一个行代表user，列代表question的矩阵，R（i，j）代表用户I对问题J的打分，这个打分可以是0，1代表有没有浏览，可以是花费的时间，也可以是星级称赞。然后用此matrix里的每一行作为对应问题的向量表示，带入上面类似的模型，计算dist，排序
Tom：确认了一下模型问，下一步怎么improve
我：（继续开始开脑洞。。。）想办法把content和collaborative filter结合起来，用一个stack denoise autoencode 去learn 问题的encoding，代表内容的信息，同时随机生成一个trainable的userembedding，把两个向量点乘和对应的rating做RMSE代表collaborative filtering的loss，同时把autoencode的loss 加在一起形成total loss，用SGD同时优化两个图模型
Tom：（可能我表述的不太清楚，这一部分花费了挺长时间反复确认相互理解对方的意思）

Part4：
Tom：多次提到了word2Vec模型，能不简要介绍这个model怎么运行的
我：（这个确实准备的时候没有复习到，完全凭借对NLP课上作业的记忆，基本上应该算大上来了）用一个词的one-hot encoding作为input，全连接到一个linear的hidden layer，然后直接输入去预测前后几个词的one-hot最后优化cross-entropy loss。训练完成后hidden layer的matrix就是结果，每一行对应了一个词的encoding。（应该大体没错，要是这个没对的话，感觉自己像个SB一样）
Tom：还有5分钟时间，有什么问题吗
我：BLABLABLA
TOM:BLABLABLA

---

## 4. https://www.1point3acres.com/bbs/thread-417455-1-1.html

第一个是个中国人，用中文面的，问了一些基本的 ML问题，什么是bagging，randomforrest和普通bagging的区别。然后是code题，就是leetcode里的wordbreak原题，但楼主太菜了，没刷到，所以只会用最笨的recursive的方法做，然后还没加memoization，时间复杂度也不会求，后来题改成了输出所有的可能的combination，更不会了，当时估计就跪了，想着还是回去还是好好刷题吧。面试官直接问你到底上过数据结构吗，直接想钻桌子了。
. check 1point3acres for more.
第二个是个美国人，问了一些project，然后说给了M行数据和N个features，然后建一个decision tree，target是房子的价格，features包括#of bedroom, # of bathroom. etc 针对某一个feature写一个split然后找到应该在哪里做split。楼主前一天刚看了information gain，还好能扯一点，然后写了一个for loop说把所有的unique variable都遍历一遍就可以了，最后又是球时间复杂度，练得少，想了半天才答上来

第三个看上去像中东人，问了应该是个DFS的问题吧。有一个m*n的table，每一个格子里有个字母，然后再给了一个prefix tree，让把所有的在prefix里的单词都找出来，每个格子在每个单词里只能用一个，我就不说自己怎么做的了，反正做的稀烂。

第四个又是中国人，还是让建一个decision tree，但这次的target是classes，所以用了entropy。. check 1point3acres for more.

第五个就随便问了一些behavior的问题，比如如果和团队大佬有分歧，或者发现了同事出错了应该怎么办。

## 5. https://www.1point3acres.com/bbs/thread-208034-1-1.html

1. 解释SVM, GBRT, CNN
2. 常用的regularization
3. cross validation相关，以及如何将cross validation和parameter tuning结合起来
4. 一些semi-supervised learning的基本概念
5. 给10000个未标注的tweets作为训练数据，和pre-defined的5个类别，如何标定数据？如何设计分类器？(crowdsourcing)
6. 当训练数据不均匀的时候如何解决？如何验证结果？(F measure，然而并没有答上来看来姿势水平还是too low)
6. 常用的聚类算法。对于kmeans，如何选择初始点？如何选择K？
7. 如何探测underfitting / overfitting？如何解决？
8. stochastic gradient descent，算法不收敛有哪些可能原因？
9. CNN parameter tuning的方法有哪些？
10. 如何计算图片相似度？


## 6. https://www.1point3acres.com/bbs/thread-395828-1-1.html

1. 几本教科书。。。（ESl, PRML或者Kevin Murphy的ML)。看比较浅的就行了，因为面试时候大概一般推不了太复杂的公司。还有andrew ng的课件和notes， 不是coursera上的，就是他自己的课的，记不得地址了。里面有个讲怎么分析模型该怎么改进的，bias vs variance, cost fucntion或者optimization方法对不对是否收敛了之类的。同时也看了学校老师的ml的课件，感觉应付基础知识够了。

http://cs229.stanford.edu/notes/cs229-notes-all/error-analysis.pdf

2. 几个基本的模型的推导，cost function等等，比如naive Bayes， linear/logistic regression, random forest/decision tree. 1point3acres

3. performance measure: 几个metric，classification/regression分别用啥等等。基本从accuracy，confusion matrix, recall/precision, F-score 到 ROC吧。multiclass classification怎么衡量。recall, precision, Fscore之类的具体是怎么算的，应该也得记住

4. unbalanced dataset 怎么处理，这个基本每家都会问。（有家startup还问了我知不知道SMOTE。。。）
. From 1point 3acres bbs
5. bias variance这个基本每家也都会问 (Random Forest到底是什么呢，难道真的说是low bias low variance?)

6. ml design: 感觉大部分时候是看feature engineering的感觉，此外就是怎么定义这个问题，是classification/regression还是unsupervised。感觉公司的话，主要还是在supervised方面来考察

7. ml coding: 感觉在半个小时时间里面只能写Kmeans或者KNN。。。 knn的话，选topK的话可以用quick selection 达到 O(n) 来代替sort或者heap。或者就是distance measure的变种。。。比如Zillow家的cosine distance。话说我刚巧之前一个project尝试了几乎所有的distance measure。。。否则应该还不会记得cosine distance的公式怎么写。

最后有个问题就是：对于一个classification问题，如果training set里面 label是有错误，改怎么处理。我不知道到底该怎么解决，一般说出来的方法是，就是train一个带probability的model(比如logistic regression)，然后根据预测的结果，两边都选probability高的，重新train。 （潜在缺点是，可能全选了错label的，所以可能需要多次random sampling

## 7. https://www.1point3acres.com/bbs/thread-323360-1-1.html

inkedIn machine learning intern 电面一面面经：
首先问了简历上education background和项目经历。
然后是coding部分：
Given: random01Fair() -- returns integers 0 or 1, with equal probability。Need to do: random06Fair(), returns integers [0,6]  with equal probability。方法：利用三个random01Fair function得到长度为3的0,1 random sequence，转化为0-7内random number，如果结果在0-6内，直接return，如果得到7，则继续调用random01Fair，知道得到0-6内random number.
然后是machine learning knowledge:
让我选择从linear regression或logistic regression中选一个解释。我选择了logistic regression，问了表达式，如何定义loss function，为啥不能用squared loss而需要用cross entropy(不是convex函数)。然后问了如果classification data is skewed，如何处理（通过调整logistic regression中的threshold）。然后问了如何validate threshold的value（Using AOC curve to validate）

## 8. https://www.1point3acres.com/bbs/thread-297793-1-1.html

2. ML，logisitic regression, cost函数是什么，什么是overfitting，如何detect，如何避免，tree 的overfitting， 有哪些distance metriccs， pvalue是什么，kemans 算法具体怎么做的， 如何找outlier， recommendation system，similarity metircs

## 9. https://www.1point3acres.com/bbs/thread-207323-1-1.html



接下来是半小时的ml questions，问了大概有logistic regression, decision tree, random forest, k means, feature selection(exploratory data analysis, l1 regularization, dimension reduction such as PCA), cross validation, overfiiting(l2 regularization)。然后让楼主挑一个ml model详细说说，挑了logistic regression，objective function + numerical methods to solve (gradient descent, stochastic gradient descent, newton method)。总体感觉都是ml basic questions


1.        简化版KNN算法写code实现，用Euclidean distance。（heap tree那里我是用heaplist实现的，出了个bug，而且写得太慢把时间耗光了）
2.        概率题，扔硬币，n次中有m次head，问是否能说明硬币是biased，写个函数算出来（经典题不难，但是没时间了没做完。）
已经挂了。。。。Feedback是要做完两道题才能过。


电面：把binary tree zigzag 相连；ML basics, 怎样处理overfitting / underfitting; 怎么从free text提取skill (面试官想要的应该是conditional random fields, 我对neural nets最熟就说了类CRF的neural net结构，他说”very interesting”)

onsite
一轮：纯coding. Roman2Int, Int2Roman, Insert intervals (最后细节没有处理好，没写完)
二轮：data mining coding. implement K-means, sparse matrix operations
三轮：host manager. 这轮很杂，从概率、ML basics、ads targeting算法设计到culture fit都有。
四轮：machine learning theory. 国人小哥用中文面的，感觉面的非常不顺。问了logistic regression的objective function, MLE推导，softmax的细节，什么是overfitting，如何防止overfitting，l1 / l2的区别，怎么从Laplacian / Gaussian priors推出l1 / l2。蛋疼的是小哥对理论解释和推导非常执着，对于常见的直观解释（画图举例）不满意，不断问“高维怎么办？”、“l2 penalize weights为什么就能防止overfitting吗？不是所有weights都被penalize了吗？”这类问题。。。好多答得不好，面完后觉得自己挂定了。
五轮：老墨+国人大哥，system design；有用户、工作和很多在线课程。问如何给用户推荐课程，使得他们上完课更容易得到那份工作。 面完后俩人sync了一下，国人大哥把我送出了公司，路上跟我说：“他说你一开始表现不好，我和他争辩了一下，这轮反正让你过了。”听得我蜜汁感动。。。



电面：大概一小时吧先是各种基础知识，logistics regression的目标函数，L1， L2 regularizer， cross validation，SVM，然后问了我一下，怎么把SVM的output按照概率输出，这个当时我没答出来，但是我瞎说了一个，我说假设svm的residual按照高斯分布，然后我们训练的时候多训练一个高斯模型好了……并且当时我跟他讲，我这个是自己瞎想的……事后发现店面的面试官没挂我，而且其实我答得也不算特别特别的离谱，只不过把我的高斯模型换成logistic regression就可以了

上门：
哎，我的上门真的是惨不忍赌啊，11月29号上门的吧，很奇怪，别人一般是5轮，我那天是四轮。

第一轮，coding：
LC 留酒吧，这题答得不好，我是看过答案的，所以上来就啪啪啪啪一顿瞎写，面试官直接说了，你少问了我一个问题，导致你的答案是sub optimal……你为啥不问我我的数列是全正还是可以有正有负？估计被人家看出来我做过这题了
第二题，sqrt，newton的方法不记得了，面试官想提醒我怎么写牛顿，但是我还是没写出来，最后写的binary search

第二轮：机器学习基础：
以前的面经上全有，而且没怎么改，我噼里啪啦一顿把所有的问题都答出来了，而且能推得我全部推了一下，感觉面试官还是比较满意的，最后时间还有，叫我推了一下神经网络的back propagation，然后时间还有，有问了我一到西雅图下雨几率的题目，是个facebook面试的高频题，glassdoor上能查到这题， 用贝叶斯做的，我当时没做过，但是自己磕磕绊绊的写的差不多了吧……

第三轮：machine learning coding：
这轮是最惨的，两位中国大哥，以为主面，一位是shadowing。有自己没准备好的原因，也有临时意外的原因，因为以往的鸡精，这轮是考想KNN啊，naive bayes的implementation的，就是直接写这些算法
先来了一个，求一个convex数组的最小值，其实就是binary search了，但是对各种corner case要求挺高，我答得也不是完美的
然后第二题，主面说，给你一个数列，代表各个点的坐标，求到所有点距离最近的坐标。而且他说，你证明一下。我去，我开始啪啪啪写公式想证明，结果我公式里面有错，得出的结果是mean of the array，大哥说不对，是median，然后他给我直观的证明了一下，我当时就晕了，跟我想的证明不一样啊！然后叫我写median，这块就怪我了，没好好刷题，quick select算法不会写，heap大哥又觉得不好，我写来写去quick select写的也不好

第四轮，HM面：
聊了一半的BQ，后面是一个machine learning设计题：
说，有一个job，有一些requirement，然后呢，一个人，有一些skills，让我设计一个算法，给这个人推荐一些课程，使他更好地met这个job的要求。
这轮是真瞎啊，我至今想不到这个设计应该怎么做……当时我到最后也瞎了，就说，对这个job 的要求，和这个人的skills和所有的课程做类似于word embeding的东西，能看出来HM的那个失望啊……

问了clustering算法有哪些？知道kmeans吧，选择kmeans的前提条件是什么，相当于对data做了什么假设啊？怎么选择k？怎么scale data之类的？然后又问classification跟clustering有什么区别？总体来说ml很基础的问题。

Quora家ML Engineer一面是ML知识考察，不涉及编程。至少我的邮件里是这么说的。。面试体验挺好，感觉是目前最喜欢的面法。
面试官简单介绍了一下自己并让我简单介绍自己以后就开始提问了。提问方法是提出若干scenario，然后让你设计解决办法，并在此过程中疯狂追问涉及到的model。
举例：面试官提问如果区分good comment和bad comment。我回答了这是一个binary classification问题并提了几个model (logistic regression, SVM)。然后面试官开始疯狂追问，问的非常细。比如logistic regression的公式？怎么train？gradient descent怎么做？有几种gradient descent呀，你会怎么选择？一定能收敛到全局最优吗？learning rate怎么选？问完以后开始问SVM，也是类似的问法。详细介绍SVM？比较SVM和logistic regression？后来还提到了word2vec，于是面试官又借机狂问了一通neural network。反正就是穷追猛打，稍有涉及的都要问个一清二楚，但并没问什么很复杂的model（可能我也没提什么很复杂的model吧==）。个人感觉是自己提到的东西一定要非常清楚，因为面试官会详细问到每个vector代表啥，vector里的值怎么取。


第一轮没有任何coding。先是互相介绍，然后面试官拿了一个list的ml题开始挨个问。我记得的有这些：
1. Binomial distribution的共轭先验
2. 假如能分开train，convolution和fully connected layer哪个用cpu哪个用gpu
3. 描述两个数据降维方法
4. 描述两个clustering算法
5. 说出三个neural netowork的regularization方法
6. 给了一个case（后来发现是以前实习生的项目==），要求大概描述一下怎么解决
7. 讲一个做过的project，要求包含提升poor model的过程


尤其是如何做feature selection问了很久。接下来，面试官让我go through之前做过的data challenge，基本上就是会问你，这一步为什么要这么做，为什么不可以那样那样那样...举个例子，EDA哪些变量需要注意，阐述一下RF算法，解释和GBM的区别，如何判断overfitting这些。感觉基本上之前做data challenge想好了问题都不难。总的来说，没有考coding，可能之后还会有下一轮电面加虾图的onsite。


4.     Machine     Learning Interview
面试官以为自己是需要面 coding，来了才发现是面 ML，只得临时准备，只好问我自己的research。中间间断问了一下 ML的基础知识，比如什么是 AUC，Bagging 等。

5.     Machine     Learning Interview
先说什么是 K-Means
Implement K-Means
If you have multiple machines, how do youimplements distributed version of k-means.
