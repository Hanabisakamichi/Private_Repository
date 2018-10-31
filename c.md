


tcp 三次握手
http://www.52im.net/thread-513-1-1.html
http://www.52im.net/thread-515-1-1.html

数组和链表区别
数组的优点：随机访问性强，查找速度快
数组的缺点：插入和删除效率低，可能浪费内存，内存空间要求高，必须有足够的连续内存空间，数组大小固定，不能动态拓展
链表的优点：插入删除速度快，内存利用率高，不会浪费内存，大小没有固定，拓展很灵活
链表的缺点：不能随机查找，必须从第一个开始遍历，查找效率低

线程进程区别
1）进程是资源分配的最小单位，线程是程序执行的最小单位。
2）进程有自己的独立地址空间，每启动一个进程，系统就会为它分配地址空间，建立数据表来维护代码段、堆栈段和数据段，这种操作非常昂贵。而线程是共享进程中的数据的，使用相同的地址空间，因此CPU切换一个线程的花费远比进程要小很多，同时创建一个线程的开销也比进程要小很多。
3）线程之间的通信更方便，同一进程下的线程共享全局变量、静态变量等数据，而进程之间的通信需要以进程间通信（IPC)的方式进行。不过如何处理好同步与互斥是编写多线程程序的难点。
4）多进程程序更健壮，多线程程序只要有一个线程死掉，整个进程也死掉了，而一个进程死掉并不会对另外一个进程造成影响，因为进程有自己独立的地址空间。

多线程的实现方式

C++：
虚函数和多态：
  多态：“一个接口，多种方法”
  静态多态：函数重载
  动态多态：虚函数
  动态联编：在使用指针访问虚函数时，通过指针指向的对象（而不是指针类型）来决定所要调用的函数 两个条件：1)虚函数，2）通过指针或引用调用虚函数
  纯虚函数：没有给出有意义的实现的虚函数，含有纯虚函数的类成为抽象类，不能声明对象（实例化），只能作为派生类的基类
  
  
和纯虚函数区别
集合类的区别，
比如list和vector区别，
map和哈希的区别


常用的linux操作


https://www.nowcoder.com/search?type=post&order=time&query=%E5%AE%9C%E4%BF%A1&page=2
https://blog.csdn.net/a2011480169/article/details/74370184
https://www.cnblogs.com/stubborn412/p/4033651.html
https://blog.csdn.net/u010358168/article/details/78785093
https://blog.csdn.net/sinat_29214327/article/details/80686992


1.4 friend
两张图：
h1，h2 起muses  测h1
h1 python run_sender
h2 python run_receiver
veth2，h3 起muses 测veth2
h1，h2用完都要kill
两个测量数据作图  用1.1的tput_draw  不用delay

h1，h2 起cubic 测h1
h1 python sender
h2 python receiver
veth2，h3 起muses 测veth2
h1，h2用完都要kill
两个测量数据作图  用1.1的tput_draw  不用delay

总时间？

1.5 背景流
一张图
h1,h2 起traffic genertor 测h1端口
bandwidth－h1端口数据 ＝ optimal
veth2 测量八种算法
veth2测量数据与optimal作图，用1.1的tput_draw  不用delay




https://fantiga.com/20170816-%E4%B8%80%E4%B8%AA%E4%BA%BA%E7%9A%84%E6%97%A5%E6%9C%AC%E8%A7%82%E5%85%89%E4%B9%8B%E6%97%85-1-%E2%80%94%E2%80%94%E5%B0%8F%E6%A8%BD%E9%9B%AA%E5%9B%BD%E7%AB%A5%E8%AF%9D-42P.html



github fork一个分之后，过一段时间就会和主分支的差异比较大。 这样提交pr的时候就会冲突，这个时候我们就需要和主分支同步代码。

步骤：
0. 在本地git仓库目录下

1. git remote add upstream git@github.com:coreos/etcd.git   //本地添加远程主分支，叫upstream。可以先git branch -v查看是否已添加远程分支，若已添加，该步骤略过。

2. git fetch upstream  // 获取主分支的最新修改到本地；

3. 将upstream分支修改内容merge到本地个人分支，该步骤分成2步：

    1） # git checkout 分支名；  // checkout到某分支

    2） # git merge upstream/分支名；  //合并主分支修改到本地分支；

4. git push                                // 将本地修改提交到github上的个人分支

至此，主分支修改的代码完全同步到fork出来的个人分支上，后续在个人分支上修改提交pr时就不会冲突。


指数基金

dog250

https://fantiga.com/categories/%E8%A1%8C%E8%A1%8C%E6%91%84%E6%91%84/
https://pantheon.stanford.edu/measurements/cloud/
https://github.com/StanfordSNR/pantheon 
https://www.jiqizhixin.com/articles/2018-08-28-5  多巴胺

https://www.zhihu.com/topic/20070859/top-answers    GAN
http://baijiahao.baidu.com/s?id=1602795888204860650&wfr=spider&for=pc k8s
https://www.zhihu.com/question/35067324 招聘
https://zhuanlan.zhihu.com/p/25298020  DRL
https://zhuanlan.zhihu.com/p/25319023  DRL
https://zhuanlan.zhihu.com/p/25328686  ML
https://zhuanlan.zhihu.com/p/27967531  ML
https://zhuanlan.zhihu.com/chuchu      AI
https://blog.csdn.net/wuzlun/article/details/80053277  Matplotlib
http://www.52im.net/thread-513-1-1.html  tcp
https://baijiahao.baidu.com/s?id=1595183355817322735&wfr=spider&for=pc numpy matplotlib pandas
https://www.cnblogs.com/dev-liu/p/pandas_plt_basic.html pandas matplotlib
https://www.cnblogs.com/zzhzhao/p/5269217.html py



www.datacamp.com



http://code.huawei.com/y84107158/AI-CC/blob/master/Indigo_Training/dagger/worker.py
http://office.huawei.com/sites/2012-csrussell/protocol/SitePages/WelcomePage.aspx?FollowSite=1&SiteName=%E5%8D%8F%E8%AE%AE%E5%AE%9E%E9%AA%8C%E5%AE%A4

lstm
https://blog.csdn.net/Prodigy_An/article/details/52832105
http://colah.github.io/posts/2015-08-Understanding-LSTMs/
http://deeplearning.net/tutorial/lstm.html

rl
https://blog.csdn.net/jinzhuojun/article/details/80417179
https://blog.csdn.net/jinzhuojun/article/details/78007628
https://github.com/openai/baselines/blob/master/baselines/ppo2/run_atari.py
https://blog.csdn.net/weixin_38195506/article/details/75550438

rnn
https://blog.csdn.net/UESTC_C2_403/article/details/73353145
https://blog.csdn.net/sydpz1987/article/details/51340277
https://www.leiphone.com/news/201709/QJAIUzp0LAgkF45J.html
https://zhuanlan.zhihu.com/p/28054589
http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/recurrent.html

tcp
https://blog.csdn.net/xiaoshengqdlg/article/details/23349591
https://blog.csdn.net/u010643777/article/details/78702577
https://blog.csdn.net/ebay/article/details/76252481
https://blog.csdn.net/dog250/article/details/52830576
https://blog.csdn.net/jtracydy/article/details/52366461
https://www.cnblogs.com/fll/archive/2008/06/10/1217013.html
http://www.52im.net/thread-515-1-1.html
http://www.52im.net/thread-513-1-1.html
https://blog.csdn.net/dog250/article/details/53013410

python
https://www.cnblogs.com/zzhzhao/p/5269217.html
https://www.cnblogs.com/dev-liu/p/pandas_plt_basic.html
https://blog.csdn.net/genome_denovo/article/details/78118511
https://baijiahao.baidu.com/s?id=1595183355817322735&wfr=spider&for=pc
https://blog.csdn.net/wuzlun/article/details/80053277

nk
https://www.nowcoder.com/activity/oj
https://www.nowcoder.com/practice/f836b2c43afc4b35ad6adc41ec941dba?tpId=13&tqId=11178&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tPage=2
http://bbs.yingjiesheng.com/forum.php?mod=forumdisplay&fid=1943&typeid=7554
http://bbs.yingjiesheng.com/forum-62-1.html
http://bbs.yingjiesheng.com/thread-2158731-1-1.html


il
http://www.sohu.com/a/219590712_129720
https://zhuanlan.zhihu.com/chuchu
https://zhuanlan.zhihu.com/p/27967531
https://zhuanlan.zhihu.com/p/25328686
https://zhuanlan.zhihu.com/p/25688750
https://zhuanlan.zhihu.com/p/25319023
https://zhuanlan.zhihu.com/p/25298020
http://www.sohu.com/a/219644723_494939
https://www.cnblogs.com/wangxiaocvpr/p/8016414.html
https://blog.csdn.net/sysstc/article/details/76214579


https://blog.csdn.net/sysstc/article/details/76214579	Linux常用命令大全（非常全！！！）
https://imlogm.github.io/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/vgg-complexity/  以VGG为例，分析深度网络的计算量和参数量
https://blog.csdn.net/u012968002/article/details/72331251 	深度神经网络-权值参数个数计算

https://blog.csdn.net/weixinhum/article/details/79273480 	深度学习1---最简单的全连接神经网络
http://www.cnblogs.com/chuxiuhong/p/5885073.html 	正则表达式入门
https://www.cnblogs.com/fanweibin/p/5053328.html 	socket
https://blog.csdn.net/yuehailin/article/details/68961304	白话经典算法系列之五 归并排序的实现
https://www.cnblogs.com/yeayee/p/4952022.html	Python 一篇学会多线程


