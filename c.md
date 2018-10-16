


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
