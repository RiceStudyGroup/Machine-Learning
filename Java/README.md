## String, string builder and string buffer

String is not mutable, so it's thread safe.

String builder is not thread safe, so it's faster.
String buffer is thread safe.

https://stackoverflow.com/questions/2971315/string-stringbuffer-and-stringbuilder

![](https://ws2.sinaimg.cn/large/006tNc79ly1fzu10femstj312w0iygr1.jpg)

## what is overload,

same signature but different arguments.

## what is the difference between == and equals?

== 判断两个变量是否引用同一个对象，而 equals() 判断引用的对象是否等价。

== return if two instance's address are same,
while equal is a method of object, and you can override this method.
for example, maybe two user instances are not same instance.
a = new User(name = ll, age = 11)
b = new User(name = ll, age = 11)
so a != b, but a.equals(b) might be true.


---
