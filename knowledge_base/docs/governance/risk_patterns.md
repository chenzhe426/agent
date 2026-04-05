# 语义风险模式参考

本文档列出 Layer 3 语义风险检测的所有模式（中英文双语）。

## 1. 指令覆盖 (instruction_override)

试图覆盖或忽略系统指令。

**风险等级**: HIGH

### 英文模式
```
ignore (all) previous instructions
disregard (all) (your) instructions
forget (all) (your) instructions
you are now
you are a
new instructions?:
system prompt hack
prompt injection
override your
bypass your (safety|content policy)
ignore your (safety|content policy)
pretend you are
as a (large) language model
no restrictions
without any (ethical|safety)
```

### 中文模式
```
忽略之前.*指令
无视.*指令
忘记.*规定
你现在是
你是一个
新的指令[是为]?:?
系统提示.*攻击
提示词.*注入
绕过.*安全
假装你是一个
没有.*限制
不顾.*伦理
不要遵守.*规则
违反.*规定
```

---

## 2. 角色提升 (role_escalation)

试图获取更高权限或扮演管理员角色。

**风险等级**: MEDIUM

### 英文模式
```
as an admin
as an administrator
with admin privileges
you have root access
bypass (security|authentication)
disable (security|firewall)
grant (full|root) access
you are in debug mode
```

### 中文模式
```
作为.*管理员
以.*管理员
拥有.*权限
绕过.*安全
关闭.*验证
获取.*root
调试模式
提权
取得.*控制权
```

---

## 3. 数据泄露 (data_exfiltration)

试图提取或窃取敏感数据。

**风险等级**: HIGH

### 英文模式
```
show me all passwords
dump (the) database
extract all user data
export (all) sensitive data
show me (the) secrets?
access (all) private (data|information)
retrieve (all) credentials
decrypt (all) passwords
hack into
steal (user) data
password list
api keys?
access tokens?
secret (keys?|tokens?)
```

### 中文模式
```
显示.*密码
导出.*密码
获取.*密钥
导出.*数据库
泄露.*用户数据
导出.*敏感信息
窃取.*数据
黑客.*进入
密码列表
API.*密钥
访问.*令牌
私钥
```

---

## 4. 未授权持久化 (unauthorized_persistence)

试图在会话外保持状态或修改系统提示。

**风险等级**: MEDIUM

### 英文模式
```
remember (this) forever
always do
never forget
store permanently
save to (your) long[-_]term memory
persist across sessions
make this permanent
update (your) base prompt
change (your) (system) instructions? to
add to (your) (system) prompt
```

### 中文模式
```
永远记住
永久保存
始终.*执行
不要忘记
长期存储
跨会话.*持久化
修改.*基础.*指令
改变.*系统.*提示
更新.*系统.*提示词
```

---

## 5. 路由操纵 (routing_manipulation)

试图操纵路由或 Agent 选择。

**风险等级**: HIGH

### 英文模式
```
route to (an) admin agent
use (tool|function) X instead of Y
skip the supervisor
bypass (the) routing
directly call
force agent
override (the) router
manually select agent
ignore task type
force task type to
```

### 中文模式
```
路由到.*管理员
使用.*工具.*而不是
跳过.*监督
绕过.*路由
直接调用
强制.*代理
手动选择.*代理
忽略.*任务类型
强制.*任务类型
```

---

## 6. 跨会话污染 (cross_session_pollution)

试图操纵其他会话的数据或状态。

**风险等级**: MEDIUM

### 英文模式
```
in session [a-zA-Z0-9\-]+
for session [a-zA-Z0-9\-]+
across (other) sessions?
in another (user) session
session hijack
poison (other) session
affect (other) users?
modify (other) users?['\s]?\s*(data|queries?|context)
```

### 中文模式
```
在.*会话中
对于.*会话
跨.*会话
另一个.*会话
会话.*劫持
污染.*会话
影响.*其他.*用户
修改.*其他.*用户.*数据
在.*session[_-]?\d+
```

---

## 7. 工具/动作 Priming (tool_action_priming)

试图预设工具使用或绕过正常流程。

**风险等级**: MEDIUM

### 英文模式
```
use (only) tool
only use (the)
you must use tool
instead of (searching|querying)
don't search just
skip search and directly
avoid using (tool|function)
only answer from
ignore the context and
bypass RAG and
```

### 中文模式
```
只使用.*工具
必须使用.*工具
不要搜索.*直接
跳过搜索.*直接
不要使用.*工具
只从.*回答
忽略.*上下文.*直接
绕过.*检索.*直接
```

---

## 8. 意图劫持 (intent_hijacking)

试图劫持或转移原始用户意图。

**风险等级**: MEDIUM

### 英文模式
```
actually,? let's
forget (about) that and
instead,? (let's)?
change the subject
new topic:
on (a) second thought
wait, (actually|forget)
let's do something else
switch to
move to topic
```

### 中文模式
```
实际上.*我们
不如.*我们
算了.*我们
换个话题
新话题[是为]?:?
等等.*实际上
让我们.*做点别的
切换到
转到.*话题
```

---

## 9. 上下文污染 (context_pollution)

试图注入伪造或恶意的上下文信息。

**风险等级**: MEDIUM

### 英文模式
```
the user said tomorrow that
pretend the date is
the user is in
user's (real) (name|location|address) is
inject (fake|falsified) data
the following is true [:：]
trust this new information
additional context [:：]
hidden prompt [:：]
```

### 中文模式
```
用户.*说明天
假设.*日期
用户.*位于
用户的.*真实.*是
注入.*虚假.*数据
以下.*是真实的
信任.*新.*信息
附加.*上下文
隐藏.*提示
```

---

## 模式匹配优先级

1. **Layer 1 (注入)**: XSS/HTML/代码注入 → CRITICAL
2. **Layer 3 (语义)**:
   - instruction_override / data_exfiltration / routing_manipulation → HIGH
   - 其他 6 类 → MEDIUM

## 注意事项

- 所有模式默认使用 `re.IGNORECASE | re.DOTALL` 标志
- 模式按照优先级顺序匹配，第一个匹配即返回
- 建议定期更新模式库以应对新型攻击
