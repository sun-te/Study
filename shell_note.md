# Linux使用日记

## Ubuntu笔记

### 安装package
```command
apt update
apt install -y <package name>
```

## 这是一个shell的笔记

### shell 里面不得不注意的坑

#### [空格的使用](https://www.cnblogs.com/aidata/p/11739746.html)
- 对于给变量赋值的语句类似于`a=1`等号前后都不可以有空格，否则会认为a是一个执行语句。
- `if [ $a = 1 ] `，对于条件判断语句，距离中括号一定要保留一个空格，if后面也必须有个空格
### shell 里面的基本语句
#### 保留字和保留字符
- `|`：将前面的一个语句的输出，作为后面一个语句的输入
- 1表示标准输出， 0表示标准输出， 2表示标准错误，如果使用类似2>&1就代表重定向错误输出为标准输出，在保存的时候可以保存
- `$0`:输出当前文件的名字，`$#`输出所有输入变量的个数， `$@`输出所有的输入变量，`$1`输出第一个变量。
- 大小判断：`-eq`：等于， `-ne`：不等于， `-gt`：大于， `-lt`：小于 `-ge`：大于等于， `-le`：小于等于
- 中括号： `[[ $a > 1 ]]` 效果和`[ a -gt 1 ]`相同，更多的括号使用规则，可参考[链接](https://blog.csdn.net/HappyRocking/article/details/90609554)
- operation
  -  tee
#### input & output
- 输入使用`read`,系统会等待输入
- 输出使用`echo`在输出的时候可以不用双引号，如果要输出变量需要用$var来输出

```shell
# 输出文本something
echo somthing
# 输出变量值a
a=1
echo "value a ="$a
```

#### 逻辑语句

- And: `&&`， Or `||`。
- 赋值

```shell
# 赋值一个实数,不准有空格
a=1
# 复制一个命令，空格后面一定要有空格
result= cd /home/
```
- 条件判断

```shell
a=1
[ $a = 1 ]
```

- if的使用，对于if使用，需要在最后加上一个fi来结束if语句

```shell
if [ "$YES_OR_NO" = "yes" ]; then
  echo "Good morning!"
elif [ "$YES_OR_NO" = "no" ]; then
  echo "Good afternoon!"
else
  echo "Sorry, $YES_OR_NO not recognized. Enter yes or no."
  exit 1
fi
```

- while语句

我们也可以看到如果在while后面打一个换行，那么while后面的分号可以不打

```shell
#!/bin/bash
i=1
while [ $i -le 5 ]
do
 let i++
done
```
