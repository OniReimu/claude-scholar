---
name: bug-detective
description: This skill should be used when the user asks to "debug this", "fix this error", "investigate this bug", "troubleshoot this issue", "find the problem", "something is broken", "this isn't working", "why is this failing", or reports errors/exceptions/bugs. Provides systematic debugging workflow and common error patterns.
version: 0.1.0
---

# Bug Detective

系统化的调试流程，用于排查和解决代码错误、异常和故障。提供结构化的调试方法和常见错误模式识别。

## 核心理念

调试是科学问题的解决过程，需要：
1. **理解问题** - 清晰定义症状和预期行为
2. **收集证据** - 获取错误信息、日志、堆栈跟踪
3. **形成假设** - 基于证据推断可能原因
4. **验证假设** - 通过实验确认或排除原因
5. **解决问题** - 应用修复并验证

## 调试流程

### 第一步：理解问题

在开始调试前，明确以下信息：

**必须收集的信息：**
- 错误消息的完整内容
- 错误发生的具体位置（文件名和行号）
- 复现步骤（如何触发错误）
- 预期行为 vs 实际行为
- 环境信息（操作系统、版本、依赖）

**提问模板：**
```
1. 具体的错误消息是什么？
2. 错误发生在哪个文件的哪一行？
3. 如何复现这个问题？请提供详细步骤
4. 期望的结果是什么？实际发生了什么？
5. 最近有什么改动可能引入这个问题？
```

### 第二步：分析错误类型

根据错误类型选择调试策略：

| 错误类型 | 特征 | 调试方法 |
|---------|------|---------|
| **语法错误** | 代码无法解析 | 检查语法、括号匹配、引号 |
| **导入错误** | ModuleNotFoundError | 检查模块安装、路径配置 |
| **类型错误** | TypeError | 检查数据类型、类型转换 |
| **属性错误** | AttributeError | 检查对象属性是否存在 |
| **键错误** | KeyError | 检查字典键是否存在 |
| **索引错误** | IndexError | 检查列表/数组索引范围 |
| **空指针** | NoneType/NullPointerException | 检查变量是否为 None |
| **网络错误** | ConnectionError/Timeout | 检查网络连接、URL、超时设置 |
| **权限错误** | PermissionError | 检查文件权限、用户权限 |
| **资源错误** | FileNotFoundError | 检查文件路径是否存在 |

### 第三步：定位问题源头

使用以下方法定位问题：

**1. 二分法定位**
- 注释掉一半代码，检查问题是否仍然存在
- 逐步缩小范围直到找到问题代码

**2. 日志追踪**
- 在关键位置添加 print/logging 语句
- 追踪变量值的变化
- 确认代码执行路径

**3. 断点调试**
- 使用调试器的断点功能
- 单步执行代码
- 检查变量状态

**4. 堆栈跟踪分析**
- 从错误消息中的堆栈跟踪找到调用链
- 确定错误发生的直接原因
- 追溯到根本原因

### 第四步：形成假设并验证

**假设框架：**
```
假设：[问题描述]导致[错误现象]

验证步骤：
1. [验证方法1]
2. [验证方法2]

预期结果：
- 如果假设正确：[预期现象]
- 如果假设错误：[预期现象]
```

### 第五步：应用修复

修复后必须验证：
1. 原始错误已解决
2. 没有引入新的错误
3. 相关功能仍正常工作
4. 添加测试防止回归

## Python 常见错误模式

### 1. 缩进错误
### 2. 可变默认参数
### 3. 循环中的闭包问题
### 4. 修改正在迭代的列表
### 5. 字符串比较使用 is
### 6. 忘记调用 super().__init__()

## JavaScript/TypeScript 常见错误模式

### 1. this 绑定问题
### 2. 异步错误处理
### 3. 对象引用比较

## Bash/Zsh 常见错误模式

### 1. 空格问题

```bash
# ❌ 赋值不能有空格
name = "John"  # 错误：尝试运行 name 命令

# ✅ 正确的赋值
name="John"

# ❌ 条件测试缺少空格
if[$name -eq 1]; then  # 错误

# ✅ 正确
if [ $name -eq 1 ]; then
```

### 2. 引号问题

```bash
# ❌ 单引号内变量不展开
echo 'The value is $var'  # 输出: The value is $var

# ✅ 使用双引号
echo "The value is $var"  # 输出: The value is actual_value

# ❌ 命令替换使用反引号（易混淆）
result=`command`

# ✅ 使用 $()
result=$(command)
```

### 3. 未引用的变量

```bash
# ❌ 变量未引用，空值会导致错误
rm -rf $dir/*  # 如果 dir 为空，会删除当前目录所有文件

# ✅ 始终引用变量
[ -n "$dir" ] && rm -rf "$dir"/*

# 或使用 set -u 防止未定义变量
set -u  # 或 set -o nounset
```

### 4. 循环中的变量作用域

```bash
# ❌ 管道创建子 shell，外部变量不改变
cat file.txt | while read line; do
    count=$((count + 1))  # 外部 count 不会改变
done
echo "Total: $count"  # 输出 0

# ✅ 使用进程替换或重定向
while read line; do
    count=$((count + 1))
done < file.txt
echo "Total: $count"  # 正确输出
```

### 5. 数组操作

```bash
# ❌ 错误的数组访问
arr=(1 2 3)
echo $arr[1]  # 输出 1[1]

# ✅ 正确的数组访问
echo ${arr[1]}  # 输出 2
echo ${arr[@]}  # 输出所有元素
echo ${#arr[@]} # 输出数组长度
```

### 6. 字符串比较

```bash
# ❌ 使用 = 比较字符串
if [ $name = "John" ]; then  # 在某些 shell 中不是标准

# ✅ 使用 = 或 ==
if [ "$name" = "John" ]; then
if [[ "$name" == "John" ]]; then

# ❌ 数字比较使用 -eq 不是 =
if [ $age = 18 ]; then  # 错误

# ✅ 数字比较使用算术运算符
if [ $age -eq 18 ]; then
if (( age == 18 )); then
```

### 7. 命令失败继续执行

```bash
# ❌ 命令失败后继续执行
cd /nonexistent
rm file.txt  # 会删除当前目录的 file.txt

# ✅ 使用 set -e 在错误时退出
set -e  # 或 set -o errexit
cd /nonexistent  # 脚本在此处退出
rm file.txt

# 或检查命令是否成功
cd /nonexistent || exit 1
```

## 常见调试命令

### Python pdb 调试器
### Node.js inspector
### Git Bisect

### Bash 调试

```bash
# 调试模式运行脚本
bash -x script.sh  # 打印每个命令
bash -v script.sh  # 打印命令原文
bash -n script.sh  # 语法检查，不执行

# 在脚本中启用调试
set -x  # 启用命令追踪
set -v  # 启用 verbose 模式
set -e  # 错误时退出
set -u  # 未定义变量报错
set -o pipefail  # 管道中任一命令失败则失败
```

## 预防性调试

### 1. 使用类型检查
### 2. 输入验证
### 3. 防御性编程
### 4. 日志记录

## 调试检查清单

### 开始调试前
- [ ] 获取完整的错误消息
- [ ] 记录错误发生的堆栈跟踪
- [ ] 确认复现步骤
- [ ] 了解预期行为

### 调试过程中
- [ ] 检查最近的代码改动
- [ ] 使用二分法定位问题
- [ ] 添加日志追踪变量
- [ ] 验证假设

### 解决问题后
- [ ] 确认原始错误已修复
- [ ] 测试相关功能
- [ ] 添加测试防止回归
- [ ] 记录问题和解决方案

## Additional Resources

### Reference Files

For detailed debugging techniques and patterns:
- **`references/python-errors.md`** - Python 错误详解
- **`references/javascript-errors.md`** - JavaScript/TypeScript 错误详解
- **`references/shell-errors.md`** - Bash/Zsh 脚本错误详解
- **`references/debugging-tools.md`** - 调试工具使用指南
- **`references/common-patterns.md`** - 常见错误模式

### Example Files

Working debugging examples:
- **`examples/debugging-workflow.py`** - 完整调试流程示例
- **`examples/error-handling-patterns.py`** - 错误处理模式
- **`examples/debugging-workflow.sh`** - Shell 脚本调试示例
