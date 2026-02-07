# ComfyUI Agent 架构讨论记录

## 一、项目背景与目标

### 1.1 项目定位
构建一个 Python 智能体（Agent），坐在用户和 ComfyUI 之间，理解自然语言意图，转化为 ComfyUI 工作流操作。

### 1.2 服务器环境
- GPU: NVIDIA RTX 5090 (32GB VRAM)
- Python: 3.12.3 (Miniconda) + PyTorch 2.8.0 + CUDA 13.0
- Node.js: v22.19.0
- ComfyUI: v0.11.1，运行在 127.0.0.1:6006，已安装 106 个自定义节点
- zealman-app: 已有的 Node.js/React 管理前端

### 1.3 核心需求
- 通过自然语言对话控制 ComfyUI 生成图片/视频
- 工作流创建、验证、调试、优化
- 节点和模型的智能推荐
- 执行监控和结果获取

## 二、调研结果摘要

### 2.1 ComfyUI API 能力
ComfyUI 提供完整的 REST + WebSocket API：
- POST /api/prompt — 提交工作流执行
- GET /api/object_info — 获取所有节点定义（输入/输出/参数）
- GET /api/queue — 查看队列状态
- GET /api/history/{id} — 获取执行历史和输出
- POST /api/interrupt — 中断执行
- GET /api/system_stats — 系统状态
- GET /api/models/{folder} — 列出可用模型
- WebSocket /ws — 实时推送执行进度、节点状态、预览图

### 2.2 ComfyUI 内部架构（关键发现）
- **单进程、双线程模型**：
  - 主线程：aiohttp 异步服务器（HTTP + WebSocket）
  - 工作线程：daemon thread，运行 prompt_worker 无限循环
- **执行模型**：严格串行，一次只执行一个工作流，节点按拓扑排序顺序执行
- **自定义节点**：通过 importlib 加载到同一进程内，共享 Python 解释器和 GPU
- **线程通信**：send_sync() 使用 loop.call_soon_threadsafe() 桥接工作线程到 asyncio 事件循环
- **可编程嵌入**：start_comfyui() 返回事件循环，允许嵌入到更大的应用中

### 2.3 已有项目分析

#### ComfyUI-Copilot（阿里，4500+ stars）
- ComfyUI 前端插件，依赖阿里后端 API
- 功能：工作流生成、调试、改写、调参
- 局限：不是独立 Agent，后端闭源

#### ComfyUI MCP Server 生态（9 个项目）
- joenorton/comfyui-mcp-server (185 stars, Python) — 最成熟
- nikolaibibo/claude-comfyui-mcp (6 stars, TypeScript) — Claude Desktop 专用
- peleke/comfyui-mcp (0 stars, TypeScript) — 37 工具，745 测试，最野心勃勃

#### ComfyGPT（学术项目，厦门大学+网易）
- 多 Agent 系统：FlowAgent → RefineAgent → ExecuteAgent
- 核心创新：链接表示法 [src_node, src_output, dst_node, dst_input]
- LLM 直接生成 JSON 准确率 16.8%，链接表示法 90%
- 需要微调的 Qwen2.5 模型，有可运行代码

#### OpenCode/Crush（19600 stars）
- Go 语言终端 AI 编码 Agent
- 核心模式：Agent 循环 + Tool 接口 + Session 管理
- 已支持 MCP 协议
- 借鉴价值：架构模式，非代码

#### @saintno/comfyui-sdk（241 stars，TypeScript）
- ComfyUI TypeScript 封装库
- 组件：ComfyApi / PromptBuilder / CallWrapper / ComfyPool
- 参考价值：API 封装模式

## 三、架构决策讨论

### 3.1 语言选择：Python
理由：
- ComfyUI 本身是 Python
- ComfyGPT 等学术项目是 Python
- 未来可能需要跑本地模型（transformers 生态）
- 如果选择进程内嵌入方案，必须是 Python

### 3.2 不 Fork OpenCode，借鉴模式
理由：
- OpenCode 是 Go，语言不同无法复用代码
- 核心 Agent 循环用 Python 实现不到 100 行
- 借鉴三个模式：Agent 循环、Tool 接口、Session 持久化

### 3.3 进程关系（核心架构决策，待定）
详见第四节专题讨论

### 3.4 意图分析
- 使用 Claude API 的 tool_use 能力
- LLM 自行决定调用哪个工具
- 不需要自己写 NLU/意图路由模块

## 四、进程关系专题讨论

### 4.1 方案 A：独立进程（进程间通信）

```
┌──────────────┐     HTTP/WS      ┌──────────────┐
│  ComfyUI     │ ←──────────────→ │  Agent       │
│  (已有进程)   │  127.0.0.1:6006  │  (独立进程)   │
│  Python      │                  │  Python      │
└──────────────┘                  └──────┬───────┘
                                        │ HTTPS
                                  ┌─────┴──────┐
                                  │ Claude API │
                                  └────────────┘
```

优点：
- 完全解耦，Agent 挂了不影响 ComfyUI
- 可以独立部署、独立升级
- 可以连接多个 ComfyUI 实例
- 不侵入 ComfyUI 代码

缺点：
- 只能通过 HTTP/WS API 通信，受限于 API 暴露的能力
- 无法直接访问 ComfyUI 内部状态（如 VRAM 使用详情、模型加载状态）
- 网络通信有延迟（虽然是 localhost，延迟很小）
- 需要单独管理 Agent 进程的生命周期

### 4.2 方案 B：ComfyUI 进程内嵌入（子线程/协程）

```
┌─────────────────────────────────────────┐
│              ComfyUI 进程                │
│                                          │
│  主线程: aiohttp 服务器                   │
│  工作线程: prompt_worker                  │
│  Agent 线程: agent_loop (daemon thread)  │
│    └── 直接访问 PromptServer.instance    │
│    └── 直接访问 prompt_queue             │
│    └── 直接访问 model_management         │
│                                          │
│         │ HTTPS                          │
│   ┌─────┴──────┐                         │
│   │ Claude API │                         │
│   └────────────┘                         │
└─────────────────────────────────────────┘
```

实现方式：作为 ComfyUI 自定义节点安装，在 __init__.py 中启动 daemon thread

优点：
- 直接访问 ComfyUI 所有内部状态
- 可以直接操作 prompt_queue，不经过 HTTP
- 可以访问 model_management 获取精确的 VRAM 信息
- 可以监听内部事件，不需要 WebSocket
- 零网络延迟
- 随 ComfyUI 启动/停止，无需单独管理

缺点：
- 与 ComfyUI 强耦合，ComfyUI 升级可能破坏内部 API
- Agent 崩溃可能影响 ComfyUI 稳定性
- 调试困难（在别人的进程里跑）
- 不能连接多个 ComfyUI 实例
- 受限于 ComfyUI 的 Python 环境（依赖冲突风险）

### 4.3 方案 C：混合方案（推荐讨论）

```
┌─────────────────────────────────────────┐
│              ComfyUI 进程                │
│                                          │
│  主线程: aiohttp 服务器                   │
│  工作线程: prompt_worker                  │
│  Bridge 线程: 轻量桥接层                  │
│    └── 暴露额外的内部状态 API            │
│    └── /api/agent/vram_detail            │
│    └── /api/agent/loaded_models          │
│    └── /api/agent/node_cache             │
└──────────────────┬──────────────────────┘
                   │ HTTP/WS + 扩展 API
┌──────────────────┴──────────────────────┐
│              Agent 进程                  │
│  Agent 循环 + 工具集 + Session 管理      │
│         │ HTTPS                          │
│   ┌─────┴──────┐                         │
│   │ Claude API │                         │
│   └────────────┘                         │
└─────────────────────────────────────────┘
```

思路：
- Agent 仍然是独立进程（保持解耦）
- 在 ComfyUI 里装一个轻量自定义节点作为"桥接层"
- 桥接层暴露 ComfyUI 原生 API 没有的内部信息
- Agent 通过标准 HTTP 调用桥接层 API

## 五、三层架构是否足够（待讨论）

当前三层：应用层 / 领域层 / 基础设施层

可能需要考虑的额外关注点：
- 用户接口层（CLI? Web API? zealman-app 集成?）
- 知识层（节点知识库、工作流模板库）
- 可观测性（日志、指标、追踪）

## 六、Agent 循环详细设计

### 6.1 核心循环流程
```
1. 加载会话历史 messages[]
2. 追加用户新消息
3. while iteration < MAX_ITERATIONS:
   a. 检查是否被用户取消
   b. 发送 messages[] + tools[] 给 LLM（流式）
   c. 流式接收 LLM 响应（文本片段实时显示）
   d. 保存 assistant 消息
   e. if 有工具调用:
      - 逐个执行工具（带超时、错误隔离）
      - 追加 [assistant消息, tool_results] 到历史
      - continue
   f. else: return 结果给用户
4. 超过最大迭代次数，强制停止
```

### 6.2 稳健性机制
- 最大迭代次数限制（防无限循环）
- 用户取消机制（随时可中断）
- 工具执行错误隔离（不崩掉 Agent）
- LLM API 重试（指数退避）
- ComfyUI 连接断开重连
- 会话持久化（崩溃后可恢复）
- 上下文窗口管理（防 token 溢出）

### 6.3 与 OpenCode 的对比
OpenCode 是扁平模块结构，无严格分层。核心循环模式几乎一样。
区别在于领域不同（代码编辑 vs ComfyUI 控制）和知识层的存在。

## 七、Monorepo 讨论

### 7.1 结论：现阶段不用 Monorepo
- OpenCode 也不是 Monorepo
- 项目早期边界不清晰
- 单人开发不需要多包管理的复杂度
- 用 Python package 目录分层即可

### 7.2 未来考虑 Monorepo 的时机
- comfyui-client 需要独立发布时
- 多人团队并行开发时
- 项目成熟需要独立版本管理时

## 八、最终架构方案（五层单包）

```
comfyui-agent/
├── src/comfyui_agent/
│   ├── interface/        # 接口层：CLI / API
│   ├── application/      # 应用层：Agent 循环 / Session 管理
│   ├── domain/           # 领域层：Tools / Models / Ports
│   ├── knowledge/        # 知识层：节点注册表 / 模板库
│   └── infrastructure/   # 基础设施层：ComfyUI客户端 / LLM客户端
├── tests/
├── config.yaml
└── pyproject.toml
```

进程关系：方案 A（独立进程），通过 HTTP/WS 与 ComfyUI 通信。

## 九、待决事项
- [ ] 进程关系最终确认（倾向方案 A）
- [ ] 第一版 MVP 的具体 scope
- [ ] 用户接口形态（CLI 优先还是 API 优先）
