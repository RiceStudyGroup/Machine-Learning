#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 19:33:03 2019

@author: xavier.qiu


[01]
给你一个sorted 全是整数的list，要你返回平方后的list并且依旧保持sorted
eg: [1,2,3,4] => [1,4,9,16]
数字可以为负
"""

# https://www.1point3acres.com/bbs/thread-452212-1-1.html

def square_sort(arr):
    for i in range(len(arr)):
        if arr[i] >= 0:
            break

    left = i-1
    right = i
    res = []
    print(left)
    print(right)
    while left >= 0 and right < len(arr):
        print("{} - {}".format(arr[left], arr[right]))
        if -arr[left] < arr[right]:
            res.append(arr[left] ** 2)
            left -= 1
        else:
            res.append(arr[right]**2)
            right += 1
    if left == -1:
        while right < len(arr):
            res.append(arr[right]**2)
            right += 1
    else:
        while left >= 0:
            res.append(arr[left] ** 2)
            left -= 1
    return res

arr = [-2,-1,0,1,4]
res = square_sort(arr)
res

#%%
"""

[02]

1. Find Kth smallest element in array

2. 把一个tweet中的所有链接加上HTML的<a> tag
比如:
Input: this is a twitter link http://twitter.com/twittertest
Output: this is a twitter link <a href="http://twitter.com/twittertest">http://twitter.com/twittertest</a>

1. can we just modify this array,

O(N), O(1)
or just copy it and solve in O(N), O(N)


2. CANNOT modify,

priority queue, we just keep the k small values in this pq,
time is O(NlogK),
O(K)

O(NlogN) + O(K), O(N)

"""

def find_k_smallest_value(arr, k ):
    """
    arr,
    """
    import copy
    arr = copy.deepcopy(arr)

    def partition(arr, lo, hi ):
        """
        arr[lo] is the pivot,
        hi is less than len(arr)
        """

        i = lo
        j = hi + 1
        while i < j:
            while i < hi:
                i += 1
                if arr[i] > arr[lo]:
                    break
            while j > lo:
                j -= 1
                if arr[j] < arr[lo]:
                    break
            if i >= j:
                break
            arr[i], arr[j] = arr[j], arr[i]
        arr[lo], arr[j] = arr[j] , arr[lo]
        print(arr)
        return j

    lo = 0
    hi = len(arr) - 1

    while lo < hi:
        pivot = partition(arr, lo, hi)
        if pivot == k:
            break
        if pivot > k:
            # what does pivot > k mean
            # 6 5 3 2 1 4 8
            #             i
            #           j
            # 4 5 3 2 1 6 8
            # for e, search the 3rd small
            # pivot= 6 > 3, so we move hi to be pivot - 1
            hi = pivot - 1
            pass
        else:
            lo = pivot + 1
    return arr[k]

arr = [6, 5, 3, 2, 1, 4, 8]
k = 3
# 1 2 3 4 5 6 8
# it will reutrn 3
for i in range(6):
    res = find_k_smallest_value(arr, i)
    print(res)

        #%%
"""

this is a twitter link http://twitter.com/twittertest
find any string start with http:

"""
test = "this is a twitter link http://twitter.com/twittertest"

import re
print(re.findall("http\S+", test))


words = test.split(" ")
temp = []
for w in words:
    if w.startswith("http"):
        temp.append("<a href=\"{}\">{}</a>".format(w,w))
    else:
        temp.append(w)
res = " ".join(temp)
res

# time complexity is O(N), cause we need to check each words in this
# space o(N)
# can we do it inplace? nope, cause in python, we cannot munipulate the string,


#%%

# https://www.1point3acres.com/bbs/thread-96088-1-1.html

"""
[03]

1.给一个数组, shuffle. (我记得不说非常清楚, 大概是)当前的数值的三次方是前两个数值的三次方的和.
"""



#%%
"""


3.trap rain water
以下内容需要积分高于 188 您已经可以浏览
"""
class Solution:
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int

         0 and right high is 1

         0 0 0 0 0 1, can not hold any water,
         why we need to move the left pointer,


            res += 1
        [0,1,0,2,1,0,1,3,2,1,2,1]
                       l
                       r
        lh = left high
        rh = right high

        time is O(N)
        space O(1), it will not change with N, which is the # of obstacles.

        """
        if height is None  or len(height) <= 2:
            return 0
        # if

        left_high = height[0]
        right_high = height[-1]
        left = 0
        right = len(height) - 1

        res = 0

        while left < right:
            if left_high > right_high:
                # move right pointer
                right -= 1
                if height[right] > right_high:
                    right_high = height[right]
                else:
                    res += right_high - height[right]
            else:
                # move left pointer

                left += 1
                if height[left] > left_high:
                    left_high = height[left]
                else:
                    res += left_high - height[left]

        return res



    #%%
"""
4.Find super-prime with n digit.
Super prime是一个prime, 而且所有prefix也是prime. 比如7331是super prime, 因为7, 73, 733, 7331都是prime.
方法就是brutal force search + pruning, 比如第一个digit必须是2, 3, 5, 7, 后面的digit必须是奇数.
然后实现boolean isPrime(int)函数, 参考CTCI.

https://www.geeksforgeeks.org/super-prime/

"""
# TODO do this before go back .

def isPrime(n):

    if n % 2 == 0:
        return  True
    if n % 3 == 0:
        return  True

    i = 5
    w = 2
    while i*i <= n:
        if n % i == 0:
            return  False
        i += w
        w = 6 - w
    return True


#%%
"""
5.给一堆会议室安排, 返回是否有conflict.

"""

# input is a list of tuple, or instance, intervel, it has a start and an end.

class Interval(object):
    def __init__(self, start , end):
        self.start = start
        self.end = end

def is_conflict(intervals):
    """
    args:
        intervals: a list of intervals, interval.start, interval. end
    return:
        if there are some conflicts between thess intervals


    # sort all the intervals based the start
    1,5
    6,8
    3,7
    3,9

    1,5  3,7, 3,9  6,8
    check the end of each interval is between the start and end of the next interval
    o(nlogn),
    if we cannot modify the array, O(N), reutnr false when end is be...
    else return true.
    """
    intervals.sort(key = lambda x: x.start)
    for i in range(len(intervals)-1):
        if intervals[i].end > intervals[i+1].start:
            return  False
    return True




    pass

"""



6.Python语言的问题, 包括GIL, iterator.
# this team used a lot of scala, and java

7.Machine Learning问题, SVM的原理之类.


https://leetcode.com/problems/find-duplicate-file-in-system/description/

"""

#%%

"""
第一轮的妹子非常高冷，全程拒绝和我交流，我问什么问题的时候她都说你要不先把code写了我们在讨论，体验非常不好。。
题是edit distance，经典的2维dp，我就直接说我们可以用dp来解，然后她说，你是之前见过这个题吗？……
（这可是算法课入门题啊，为啥会有人没见过）
写完之后给了一个follow up，看两个string是否能match，就是lc第十题……但是code没写完，她把代码照完相就走了。
"""


"""
第二轮的妹子好沟通一点……题很简单，从文件中读log，每条log有time stamp, user id, 和open/close app操作。
最后输出所有用户平均使用app的时间，以及每个用户单独的使用app的平均时间。
题很简单……我还以为是sql,她说是文件读写，于是就用python写了。
然后讨论了一下如果log的文件不规范怎么办，比如有的entry可能miss open/close，都是比较开放的follow up，好好讨论就可以。
"""


#%%
"""
两轮都是白板。

第三轮是纯bq。问的问题特别多。。。也是聊的过程中感觉自己不太fit他们家。
有比较经典的问题，比如你为啥学cs……
问了比如说你有哪些team work experience。然后如果我问你的teammate和你合作的strength/weakness，他们会怎么回答。
（这种negative question真的很难受）
其他的不记得了，反正我聊的很不喜欢……

https://www.1point3acres.com/bbs/thread-455675-1-1.html



"""


#%%

"""

两周前接到的HR通知，上周安排的第一轮技术电面，coding题目是找一个文档里的Top k 最频繁的单词。很简单，要么用heap，或者直接用quick selection。前者容易编写，后者最优复杂度。这周安排的2轮电面和一轮行为店面。

"""
wordlist = ["hello", "hello", "hello", "hello", "hello", "hello", "world", "world", "world", "world", "yes", "yes", "yes", "yes", "yes", "ll"]


def top_k_words(wordlist, target):
    """
    bucket sort or quick select


    """

    from collections import Counter
    c = Counter(wordlist)
    maxNum = 0
    for k in c:
        if c[k] > maxNum:
            maxNum = c[k]
    bucket = [list() for _ in range(maxNum + 1)]
    for k in c:
        bucket[c[k]].append(k)

    #print(bucket)
    start = maxNum
    res = []
    while len(res) < target:
        temp = bucket[start]
        temp.sort()
        for w in temp:
            res.append(w)
            if len(res) == target:
                break

        start-=1

    return res
res = top_k_words(wordlist, 2)
res

#%%

def top_k_words_qs(wordlist, target ):
    pass
#TODO if you want to do this again.

#%%
"""
第一轮一个伊朗人，data scientist，问了一个如何向用户推送新闻的问题，
coding题目是给定两个字符串s和t，
 请问最少用多少次swap操作可以把t变成s。swap指的是交换两个字符。

hello

hlloe



"""





 #%%
 """
第二轮一个印度人，感觉很水，leetcode两道原题，robber house和kill process。大部分时间都在提问上面了。
robber house done, don't need to this anymore, waster tiem
kill process done, not bug free, you can do this one more
https://www.1point3acres.com/bbs/thread-335238-1-1.html
"""


#%%


"""
一面 Behavior
全是behavior问题 e.g. 如果满分是10 你之前的manager给你打几分 我说8/10 然后疯狂追问 为什么不是9，为什么不是10，你觉得怎么做才能到10.。。

Emm, i guess 8 as well.

Firstly, my undergraduate study is not cs, so I spend a lot of time to learn fundamental stuffs at that period during my free time.
He is great, I mean, taught me a lot, tell me not worry to choose my direction and ..



"""

#%%


"""

二面ML Concepts
很基本的概念。讲了一些LZ学过的CNN, RNN,
但是感觉他们并不怎么用deep learning, Focus的都是比较基本的model:
random forest, vanillla NN & KNN, etc.
之后三道lc easy/medium题 秒

三面
第一题从一个树里找sum最大的path. lz写了一个dfs, 大概长这个样子。我写的是python。
def dfs(node, sum):
    \\update sum
    dfs(node.left, sum)
    dfs(node.right, sum)

    done, the int is passed by value not by reference,

竟然 被问为什么第一个recursive call完成之后，pass给第二个recursive call的sum不会变。。
然后给面试管讲了一遍function stack是怎么回事，然后python的integer是pass by value not by reference.
面试管看起来没怎么懂，感觉凉凉。

然后followup了一个变种。

面试管人都蛮好的。但是整个process感觉稍微有点坑。结果被拒了 不背锅。

https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=441772


"""


#%%


"""
https://www.1point3acres.com/bbs/thread-289874-1-1.html


中午刚面的，这题目还挺有意思。Interviewer说不考虑closed form solution, find algorithmic solution, such as recursion or iteration。
Given a jar of W white beans and R red beans.
You are hungry so you will eat all of the beans.
However, you prefer the taste of the white beans, so you will use the following rule to determine the next bean to eat:

Randomly select a bean from the jar and if it is white, eat it.
Otherwise return the bean to the jar, randomly select a bean again and eat it regardless of color.

You will repeat this process over and over again until all beans are consumed.

Write a program that returns the probability that the last (final) bean consumed is white.

You can assume that the numerical precision of floating arithmetic in your programming language is unlimited.

---


因为面的是ML engineer, 之后还问了2道design题目，类似如何根据用户query的keywords推荐广告相关的keywords, 还有一题没太理解时间就到了。。。-baidu 1point3acres



"""

"""
let's say there're W white beans and R red beans

the probability of the final bean is white


p of first bean you eat is red      = R/(W+R) * R/(W+R)
p of first bean you eat is white    = W/(W+R) + R/(W+R) * W/(W+R)


Then it becomes a sub problem.



"""

#%%

class Solution:
    def calculatePofFinalBeanRed(self, R, W):
        self.wp = 0
        self.rp = 0
        self.getProbability(R,W,1.0)


        print(self.wp + self.rp)

        return self.wp, self.rp


    def getProbability(self, R, W, rootP):
        """
        @args:
        R: the number of red beans
        W: the number of white beans
        rootP:
        @return:
        the p of eat red bean and eat white bean according to the rule
        """
        if R == 0:
            self.wp += rootP
            return

        if W == 0:
            self.rp += rootP
            return

        eatR = R/(W+R) * R/(W+R)
        eatW = 1 - eatR
        self.getProbability(R-1, W, rootP * eatR)
        self.getProbability(R, W-1, rootP * eatW)
#%%
s = Solution()
wp, rp = s.calculatePofFinalBeanRed(1,1)



#%%
"""

Twitter电面是视频面试（所以面试前还得换个衣服洗把脸梳个头啥的。。。）
面试官是个中国人上来简单介绍一下自己也没怎么问我的背景就开始做题了
总共面了两道题
第一道：给一堆用户以及其活跃时间的tuple list [<userID1, activeTime1>.....<userIDn,activeTimen>],
其中用户可以活跃于多个timestamp，在同一分钟内（e.g. 10：01：01 and 10：01：59）
活跃多次的只记为一个active minute；并且activeTime格式整数，记录了从1975.1.1到当时的毫秒数。
问题是让统计不同cumulated number of active minutes 对应的用户数量 （e.g. 总共活跃了x分钟的用户有多少个）
解法：用hashmap和set解了一波，写完程序还要运行一下。面试官follow up，问这个东西用mapreduce怎么解决。楼主有五六年没用过map reduce，凭印象讲了一下MapReduce的原理和怎么apply在这个问题上，面试官说差不多答对70%，不过念在我不咋用的份上就不深究了。。。

第二道：给一个图和source以及target找source到target的最短路径
写完程序也要拿test case运行一下出正确结果才行

面完大概过了两周多给了onsite

楼主目前正在焦虑的准备onsite，急需学习地里的面经，无奈大米不够，求各位看官赏点米！谢谢啦！
https://www.1point3acres.com/bbs/thread-445377-1-1.html

https://www.1point3acres.com/bbs/thread-459585-1-1.html
"""

# TODO here, the second one.

#%%
"""
query sum mutable

https://leetcode.com/problems/range-sum-query-mutable/description/


"""

#%%
"""
刚刚面了twitter，出了一个非常简单的题，而且做完还没有follow up，问面试官问题他也磕磕巴巴地回答。。。很快就结束了面试。。。

/**
*
* List<Iterator> itertors = [first, second, third]
*
* first: 1 2 3
* second: a, b, c. check 1point3acres for more.
* third: Ω ≈ ç
*
* new iterator:
* 1 a Ω 2 b ≈ 3 c ç
*
* // hasNext()
* // next()
*
*/

"""

#%%

"""
字符串自动匹配(就是建立一个Trie)，之前写过一次Trie，

"""


#TODO here

#%%

"""

Nested Integer [skip]
毛子面的上来就没废话直接做题
就是三四腰那道题
打印嵌套链表。。
但是要自己写testcase。这个就折腾了半天。因为lc上面的interface和testcase是给好的。
但要自定义比如getInteger，getList，怎么输入呢。。
卡了半天。。
我太菜鸡了。。

"""

#%%%


"""
1.clone graph
2.Simple Calculater， +, -, *, /无括号
3.印度老哥，问一个很有意思的数据结构skip list
4.白人小哥，是一个code review的session，大概就是一个Rental, Car, User的几个class一个非常简单的oo的小程序，考察对oo的一些认识，原来的程序把一些和Rental bussiness的logic放到user里面， 应该去refactoring，另外User里面有一个打印的class,基本就是最好定义一个interface，然后可以implement by比如 htmlprinter, simpleIOprinter, and so on. 最后一个followup 就是说如果有很多busniess rule要怎么表示，回答就是if-else vsus state machine.
5.实现insert, delete, getrandom o(1),
6.LRU
7.给一堆照片，中间有大量的重复照片，如何去重。 其实很简单就是把照片serialize成string 然后存hashmap。
8.给一个stream 的hashtag（用string表示）， 在一个size 为n的 window里求出现次数最多和最少的hashtag（不太记得了，貌似是这样的）。 基本就是queue + hashtable就行了。然后follow-up成数据很大内存放不下怎么办。
9.lc296
11.LeetCode Single Number
12.Leetcode Text Justification
13.leetcode coin-change
14.一个白人小哥，说话超快。问了查重，给你一个数组和一个数n, 数组里的数由1-n组成。现在让你找出重复的数，in-space, nlogn的时间复杂。用二分查找。先找出重复的数在1-n/2还是n/2 + 1 - n. 这样不段缩小范围，最后得到结果
15.word break
16.一个印度小哥，senior engineer，比如有字典里有｛cat, bat, pat, twitter｝,现在给你一个string s和一个integer k, 比如说“eat", 2, 让你找出所以与s的edit distance为k的string并输出来。这里就输出[cat, bat, pat].用trie数做
17.best meeting point
18.实现bigint的加法。
Coding题是吃豆子，我用的DP:
jar里有W个白豆子，R个红豆子。随机摸一个豆子。摸到白豆子直接吃。摸到红豆子，放回去，再摸一豆子，不管是红是白，都吃。问最后一个豆子是白豆子的概率。
Coding question was LC 139 Word Break but use API method boolean isFoundInDictionary(String s) instead of being given a Set<String> wordDict
有很多报纸或杂志上剪下的单词，判断能否拼成一个给定的句子用hashmap
2. lc 3 sum 是否等于target number
happy number + Valid Parentheses
给一个list of numbers 和一个block list 返回不在block list里的numbers
follow up是block list不能放在内存里怎么办
迟到了十分钟，就开始聊sytem level的问题，比如把一个很重要的文件（一个byte都不能错）从一台data server copy到另一个data server要怎么做,用scp, scp基于ssh不, ssh的加密不，为啥要加密呢. ssh是基于TCP, 既然TCP不会丢包，那文件是不是就不会损伤，我们为啥还要去check文件损坏了没有呢，用什么办法去check文件有没有损坏，MD5, 但比如把eclipse(开源）的，download到本地，也是TCP的，不会丢包，为啥要用一个MD5 code来检验这个eclipse有没有损坏呢. 文件在传输的过程中，有哪几步会有问题呢，网卡？ 内存，disk? 压缩？ 具体聊一聊在什么情况会有问题.
2. 要一个interface, 这个interface实现两个public method, 1 . public void addJob(Callable f, int time) 把一个function传进来，然后每隔time把这个function call一遍. 2.deleteJob(Callable f)停止运行这个程序，我用multi-thread写，说了下怎么用sleep来实现等待，然后keep一个deadlist,一旦这个fucntion不运行，就加到这个deadlist里，每次都创建一个新的thread来运行f. 但面试官说，这个totally find,but there is a better way just using single thread. 大家可以想一想，我想我最后就挂在这题上了..
3. 实现LRU cache，是一道考察doubly linked list的题目，蛮快就写好了
哈哈哈，因为悲剧了，所以上周没有心情发。
面试官首先了解了一下我现在在做的项目，有一些followup的behavior question，比如说在这过程中遇到的最大的挑战是啥，你是怎么克服的，blah blah blah。8 I8 X) N; w& D) V4 L2 }! A1 ]
两道问题，都是比较open的question。有很多followup， 设计算法和design。% @9 b; X$ q8 [0 y) r* Y! q( [
第一题，题目本身不难。怎么实现一个blacklist的phone book。就是用户看不到所有的电话号码，但是可以查询某个电话在不在系统里。/ ^% E: w  L; W- N
我说可以放在hashmap里，这样可以fast lookup。
followup：怎么实现电话号码的hash function, 遇到hash conflict怎么处理。
followup2: 在这个系统里，有没有multithreading的问题。没有？因为是只读。
followup3: 如果可以添加新的电话号码，怎么在multithread的环境下，protect data。

1.power set
2.maximum subarray
3.sort linkedlist
5.alien language  给定一个array 的string，如 at， cet， dog， dzg 要返回a-> e-> c-> d->o->z
6.permutation and permutation II
7.stock i ii iii 问题，就是最简单的，但是除了返回max profit之外，还要返回buy 和sell的时间复杂度要求O(n)2. check 1point3acres for more.
8.Write a fibonacci function (iterative, constant space)
9.find the lower_bound of index in a sorted string array that matches a given prefix string e.g. array=["Ann", "App",  "Apple", "Boy"], prefix = 'Ap', return 1; prefix = "c", return -1; prefix = 'B', return 2; 方法是binary search.
10.Leetcode Insert Delete GetRandom O(1)要求三个operation的时间复杂度都是O(1) 最后问了是否线程安全？,用Arraylist 和 hashset 做
11.给两个unsorted list, 找intersection, 如果是sorted list, 怎么优化, 如果有duplicated, 只返回一次，怎么改代码, 如果一个list很大，一个list很小，怎么优化
12.hashtable的存储寻找空间时间复杂度之类的东西，然后我就叽里咕噜的把各种openhash，closehash，各种解决collision的概念说了一遍
14.第二题是word break 给一个dict，一个string w/o space，问string是否能segment成dict里面出现的words
15.第一题 string shortening， 每个字母最多连续出现两次，超过两次的就不保留了"aaabaaccc--> aabaacc"
16. check string 有没有一个 anagram 是 palindrome 如果有return true 没有return false
问了leetcode two sum的改编题：given an array, return the indexes if num - num[j] = k
这个也是用hashmap就好了，但是跟sum还有一些不同。
18.implement skiplist
19.Given a list of nums, find the whether all the elems are unique
20. Hashing, Deal with hash collisions,
Largest Rectangle in Histogram
22. show tweets for common friends, (b tweets a, c can see this tweet only when a and b are both friends of c),
I used hashtable and a linear scan to find common friends.
23. Find log information for a time stamp period. (I used binary search and linear scan to find the beginning and end).. 1point3acres
Follow up: if preprocessing is allowed, how would you do it? (I used hashmap to store the timestamp, content so that log for each timestamp is a block in hashmap)
24.Word ladder leetcode
26.最简单的dp走格子那题，n*m board，从左上角到右下角，往下或往右，follow up：space optimization，space优化的时写了两个八阿哥，被教育了
27.coding题考的是求一个表达式的值，符号有＋，－，＊，没有括号。楼主答可以用stack写，不过面试官要求用recursion
28.给一个数组，对每一个元素，找出在它之前比它的第一个元素的值。如果没有比它小，则返回它 for example: input {3, 5, 2, 6, 9, 7, 10}8 E8 w& @' y! `) U: n
output {3, 3, 2, 2, 6, 6, 7} 用一个栈存递增序列，O(n)解决。
29. 给你两个有序的array， 找出它们共同的元素。
我给出了two pointers和二分查找两种方法分别在以下两种情况下：（1）两个数组的个数相当 （2）一个数组个数远大于另一个
followup：现实中，用哪种方法比较好？（two pointer）. check 1point3acres for more.
https://leetcode.com/problems/sliding-window-maximum/
follow up上面你用了extra memory来产生有序的结果，那不用extra memory怎么做？  这里没让写code，只需给idea。
这里可以观察permutation的规律，类似next permutation那道题，说了一下大概的idea，要解释清楚。



1.什么是encapsulation，优缺点
2.abstract class
3.什么是static method，有什么用途



三道题：
1） 给array of strings, 要求得出每个string里使相邻character都不相等所需修改的character个数（随意改成其他character）然后写入一个list 并 return
2 ) 给array of integer， 主题要求是找出距离每个index最近的相同value的index, 如果有多个距离相同取小的index， 如果没有相同的取-1
3 ) 给两个list of string  ： sentences, queries 要求对每个query, print出包含所有queries下单词的sentence的index，如若一个sentence能包含所有queries下单词并重复不止一次就多print几遍 index:
例 : sentence.get(i)   ->  "likes likes likes"  query->"likes"    那么就该print: i  i  i;
前两题不难， 第三题非常容易time out，截止我发post的时候还有5个case time out了，希望有好心人给些大米到150让我能看一下地里关于那题的详解。-baidu 1point3acres

谢谢各位， Good luck, fight hard.

https://www.1point3acres.com/bbs/thread-325511-1-1.html

先是来了一轮经理面

之后一轮技术面， 包括机器学习数据挖掘相关知识，然后一个coding题目 (类似finding top-k的frequent words)

后来hr发邮件说要onsite？？？ 后来发现是实习，乌龙一场，改成电面，包括:

两轮算法面，和一面BQ面

两轮算法面模式类似，都是先讲自己的pro, 然后coding

第一个coding题目，类似于给一个有向图，找length = 2的cycles

第二个coding题目，类似于给一个0/1的matrix, 找0-1的最短路长度， 用BFS给了个高复杂度的解法，复杂度之高已经让面试官蒙蔽了

BQ面的模式是manager对每一次的实习经历问你干了什么你的mentor对你什么评价你觉得你的优点是什么你的缺点是什么你的mentor认为你应该提高什么地方

BQ面没准备，答得不是很流利

算法面Twitter的好处是不是上来做题，而是先问pro, 这样就可以把自己做的pro和你申请组要做的工作做个connection。 这样可以让对方知道你的背景和这个组很match.


"""

#%%

"""
第一轮：Technical，有两位engineers。一开始先是15分钟的互相介绍，之后做了一2D matrix Zigzag Traversal，把途中经过的数字print出来即可。进入状态比较慢，最后几分钟才把完整的代码写出来。
第二轮：Lunch
第三轮：Technical，同样是对两个人，一开始也是简短的互相介绍。做了一个Design Tic-tac-Toe。其实只需要简单的做一个object oriented design，然后把里面需要的函设计好，最后再来设计算法并做优化。我把这个问题复杂化了，一开始想的是怎么做User Class，做Board Class，导致面试官也有点懵逼。这轮可能也直接决定了我最后没有拿到offer。
第四轮：Behavioral。可能是面的最舒服的一轮，能感觉manager非常愿意交流。很仔细地问了简历上的project。对于每一个实习她都问了我同一个问题：如果我有机会和你这个实习的mentor/manager沟通的话，你觉得他会说你的最大的优点和缺点是什么。


"""

#%%%


"""
Merge Timeline Tweets

Tweet:
    String content;
    Long  ts;

User:
    List<Tweets> tweets;
    List<Friends> friends;  # 这里面试的时候用的是followees

要求是 merge 这个 timelines.

for exmaple:

User Thomas:
     friends: Nancy, Jerry, etc.

Nancy: [("hhahah", 1), ("hhahah", 10)]
Jerry: [("zzzzz", 2), ("eeeeee", 3)]

要求输出 top k recent friends tweets.



"""
