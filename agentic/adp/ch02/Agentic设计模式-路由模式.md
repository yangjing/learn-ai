# Agentic 设计模式 - 路由模式（Routing Pattern）

_本文示例代码见：[routing-langgraph.py](https://github.com/yangjing/learn-ai/blob/main/agentic/adp/ch02/routing-langgraph.py)，[ch02-routing.rs](https://github.com/yangjing/learn-ai/tree/main/agentic/adp/examples/ch02-routing.rs)_

## 路由模式概述

虽然通过提示链进行顺序处理是执行确定性线性工作流的基础技术，但其适用性在需要自适应响应的场景中存在局限。现实世界中的智能体系统通常需要根据动态因素（如环境状态、用户输入或先前操作结果）在多个潜在操作间进行仲裁。这种动态决策能力通过称为路由的机制实现，它控制着流向不同专用功能、工具或子流程的路径。

路由将条件逻辑引入智能体的运行框架，使其能够从固定执行路径转向动态评估特定标准以选择后续操作的模型，从而实现更灵活、更具上下文感知能力的系统行为。例如，一个配备路由功能的客户查询处理智能体可以首先对用户查询进行分类以判断其意图，随后将其引导至专业问答代理、账户信息数据库检索工具，或是复杂问题的升级处理流程，而非局限于单一预设响应路径。

因此，采用路由的先进智能体可执行以下流程：

- 分析用户查询内容；
- 根据查询意图进行路由分配：
  - 若意图为"订单状态查询"，则路由至与订单数据库交互的子代理或工具链；
  - 若意图为"产品信息获取"，则路由至搜索产品目录的子代理或工具链；
  - 若意图为"技术支持"，则路由至访问故障排除指南或转接人工服务的处理链；
  - 若意图不明确，则路由至澄清查询的子代理或提示链。

路由模式的核心组件是执行评估并引导流程的机制，其实现方式包括：

- **基于大语言模型的路由**：通过提示让语言模型分析输入内容，并输出指定标识符或指令以指示后续步骤。例如，提示可要求模型"分析以下用户查询并仅输出分类：'订单状态'、'产品信息'、'技术支持'或'其他'"。智能体系统根据该输出引导工作流。
- **基于嵌入向量的路由**：将输入查询转换为向量嵌入，随后与代表不同路由的嵌入向量进行相似度比较，最终将查询分配至最相似的路由路径。这种方法适用于语义路由场景，其决策基于输入语义而非单纯关键词。
- **基于规则的路由**：通过预定义规则或逻辑（如 if-else 语句、switch 条件判断）根据关键词、模式或从输入中提取的结构化数据进行路由决策。该方法比 LLM 路由更快速且确定性强，但对复杂新颖输入的适应性较弱。
- **基于机器学习模型的路由**：采用经过小规模标注数据专门训练的判别式模型（如分类器）执行路由任务。虽然与嵌入方法存在概念相似性，但其核心特征是通过监督微调过程调整模型参数以创建专用路由函数。该技术与 LLM 路由的根本区别在于：决策组件并非推理时执行提示的生成式模型，而是将路由逻辑编码于微调模型的学习权重中。虽然 LLM 可能用于预处理阶段生成增强训练集的合成数据，但不参与实时路由决策。

路由机制可在智能体运行周期的多个节点实施：既可用于初始阶段的主任务分类，也可在处理链中间节点决定后续操作，或在子程序运行时从给定工具集中选择最合适的工具。

[LangChain](https://docs.langchain.com/oss/python/langchain/overview)/[LangGraph](https://docs.langchain.com/oss/python/langgraph/overview), [rig.rs](https://docs.rs/rig-core/latest/rig/) 等计算框架提供了定义和管理此类条件逻辑的显式构造。LangGraph 基于状态图架构，特别适用于决策取决于系统累积状态的复杂路由场景。同样，谷歌 ADK 提供了构建智能体能力和交互模型的基础组件，这些组件可作为实现路由逻辑的基础。在这些框架提供的执行环境中，开发者可定义可能的运行路径，以及决定计算图中节点间转换的函数或模型评估机制。

路由机制的实施使系统能够突破确定性顺序处理的限制，推动开发出更具适应性的执行流，从而对更广泛的输入和状态变化做出动态且恰当的响应。

## 实际应用与使用场景

路由模式是构建自适应智能体系统的核心控制机制，使其能够根据多变的输入和内部状态动态调整执行路径。通过提供必要的条件逻辑层，该模式在多个领域展现出关键价值。

在人机交互领域（如虚拟助手或 AI 驱动导师系统），路由被用于解析用户意图。系统通过对自然语言查询的初始分析，决定最合适的后续操作：无论是调用特定信息检索工具、转接人工客服，还是根据用户表现选择下一教学模块。这使得系统能够突破线性对话流程，实现情境化响应。

在自动化数据与文档处理流程中，路由承担分类与分发功能。系统基于内容、元数据或格式对输入数据（如电子邮件、支持工单或 API 有效载荷）进行分析，随后将每个项目引导至对应工作流，例如销售线索录入流程、针对 JSON 或 CSV 格式的特定数据转换函数，或紧急问题升级路径。

在涉及多个专用工具或智能体的复杂系统中，路由充当高层调度器。例如，由搜索、摘要和分析代理组成的研究系统，会根据当前目标通过路由器将任务分配至最合适的代理；同样地，AI 编程助手在将代码片段传递至专用工具前，会通过路由识别编程语言并判断用户意图（调试、解释或转译）。

最终，路由提供的逻辑仲裁能力，对于构建功能多样化且具备上下文感知的系统至关重要。它将智能体从执行预定义流程的静态工具，转变为能在动态条件下决策最优任务执行方式的智能系统。

## 代码示例（Python: LangGraph）

在代码中实现路由机制，需要定义可能的执行路径及决定路径选择的逻辑。LangChain/LangGraph 等框架为此提供了专用组件和架构。其中，LangGraph 基于状态图的架构能清晰直观地呈现和实现路由逻辑。

以下代码展示了使用 LangChain 与 DeepSeek 构建的简易智能体系统。该系统通过设置一个"协调器"，根据用户请求的意图（预约、信息查询或模糊意图）将请求路由至不同的模拟"子代理"处理程序。其工作原理是先用语言模型对请求进行分类，再将请求委派给相应的处理函数，模拟了多智能体架构中常见的基础委托模式。下面是完整的 Python 代码：

```python
from typing import TypedDict, Literal, cast
from dotenv import load_dotenv

from langchain_deepseek import ChatDeepSeek
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

load_dotenv()


class RouterState(TypedDict):
  request: str
  decision: str
  output: str


def create_coordinator_router_node(llm: BaseChatModel):
  """创建协调器路由节点"""
  coordinator_router_prompt = ChatPromptTemplate.from_messages(
    [
      {
        'role': 'system',
        'content': """你是一个智能助手，负责分析用户请求并确定应该将其分配给哪个专业处理程序。
    - 如果请求与预订航班或酒店相关，请输出 '预订'。
    - 对于所有其他一般信息问题，请输出 '信息'。
    - 如果请求模糊不清或不适合任何类别，请输出 '模糊'。
    仅输出一个单词：'预订'、'信息'或'模糊'。""",
      },
      {'role': 'user', 'content': '{request}'},
    ]
  )

  coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()

  def coordinator_router_node(state: RouterState) -> RouterState:
    """路由决策节点"""
    decision = coordinator_router_chain.invoke({'request': state['request']})
    return {**state, 'decision': decision.strip()}

  return coordinator_router_node


def booking_node(state: RouterState) -> RouterState:
  print('\n--- 委托给预订处理器 ---')
  request = state['request']
  output = f'预订处理器处理了请求："{request}"。结果：模拟预订操作。'
  return {**state, 'output': output}


def info_node(state: RouterState) -> RouterState:
  print('\n--- 委托给信息处理器 ---')
  request = state['request']
  output = f'信息处理器处理了请求："{request}"。结果：模拟信息检索。'
  return {**state, 'output': output}


def unclear_node(state: RouterState) -> RouterState:
  print('\n--- 处理模糊请求 ---')
  request = state['request']
  output = f'协调器无法委托请求："{request}"。请澄清。'
  return {**state, 'output': output}


DecisionType = Literal['预订', '信息', '模糊']
EXPECT_DECISION = ['预订', '信息', '模糊']


def route_decision(state: RouterState) -> DecisionType:
  """根据决策结果路由到对应的处理节点"""
  decision = state['decision']
  if decision in EXPECT_DECISION:
    return cast(DecisionType, decision)
  else:
    raise ValueError(f'无效的决策类型: {decision}，期望的类型为: {EXPECT_DECISION}')


def build(llm: BaseChatModel) -> CompiledStateGraph:
  """构建LangGraph路由图"""

  # 创建状态图
  workflow = StateGraph(RouterState)

  # 创建协调器路由节点
  coordinator_router_node = create_coordinator_router_node(llm)

  # 添加节点
  workflow.add_node('coordinator', coordinator_router_node)
  workflow.add_node('预订', booking_node)
  workflow.add_node('信息', info_node)
  workflow.add_node('模糊', unclear_node)

  # 设置入口点
  workflow.set_entry_point('coordinator')

  # 添加条件边：从coordinator根据决策路由到对应的处理节点
  workflow.add_conditional_edges('coordinator', route_decision)

  # 所有处理节点都连接到END
  workflow.add_edge('预订', END)
  workflow.add_edge('信息', END)
  workflow.add_edge('模糊', END)

  # 编译图
  return workflow.compile()


def main():
  """路由机制 - LangGraph版本"""
  llm = ChatDeepSeek(model='deepseek-chat', temperature=0)
  agent = build(llm)

  print('--- 运行**预订**请求 ---')
  result_a = agent.invoke({'request': '预订一张到重庆的机票。'})
  print(f'最终结果 A: {result_a["output"]}')

  print('\n--- 运行**信息**请求 ---')
  result_b = agent.invoke({'request': '江苏的省会是哪个城市？'})
  print(f'最终结果 B: {result_b["output"]}')

  print('\n--- 运行**模糊**请求 ---')
  result_c = agent.invoke({'request': '理讲物子量讲。'})
  print(f'最终结果 C: {result_c["output"]}')


# uv run -m agentic.adp.ch02.routing-langgraph
if __name__ == '__main__':
  main()
```

运行示例程序，输出结果如下：

```shell
$ uv run -m agentic.adp.ch02.routing-langgraph
--- 运行**预订**请求 ---

--- 委托给预订处理器 ---
最终结果 A: 预订处理器处理了请求："预订一张到重庆的机票。"。结果：模拟预订操作。

--- 运行**信息**请求 ---

--- 委托给信息处理器 ---
最终结果 B: 信息处理器处理了请求："江苏的省会是哪个城市？"。结果：模拟信息检索。

--- 运行**模糊**请求 ---

--- 处理模糊请求 ---
最终结果 C: 协调器无法委托请求："理讲物子量讲。"。请澄清。
```

## 代码示例（Rust: rig, graph-flow）

这里给出 Rust 版本的实现，相比 Python 版本添加了会话管理功能，并额外具有以下优势：

1. 类型安全 - Rust 的类型系统确保路由决策和状态转换在编译时就能得到验证
2. 内存安全 - Rust 的所有权模型可以防止并发路由场景中的内存泄漏和数据竞争
3. 性能优势 - 零成本抽象和无垃圾回收机制为路由操作提供更好的吞吐量
4. 会话管理 - 内置的会话存储功能可以在路由决策过程中维护状态（注：LangGraph 也支持会话管理）
5. 异步支持 - 一流的异步支持可以高效处理并发路由请求

_为简洁起见省略了导入语句：_

```rust
/// 协调器路由任务
struct CoordinatorRouterTask {
  agent: Agent<deepseek::CompletionModel>,
}
impl CoordinatorRouterTask {
  fn new(system_prompt: &str) -> Result<Self> {
    let client = deepseek::Client::from_env();
    let agent = client.agent("deepseek-chat").temperature(0.0).preamble(system_prompt).build();
    Ok(Self { agent })
  }
}
#[async_trait]
impl Task for CoordinatorRouterTask {
  async fn run(&self, context: Context) -> graph_flow::Result<TaskResult> {
    let request: String = context.get_sync("request").unwrap_or_default();
    let response = self
      .agent
      .prompt(&request)
      .await
      .map_err(|e| anyhow::anyhow!("Failed to get routing decision: {:?}", e))?;
    context.set("decision", response.trim()).await;
    Ok(TaskResult::new(Some(format!("路由决策: {:?}", decision)), NextAction::Continue))
  }
}

/// 预订处理任务
struct BookingTask;
#[async_trait]
impl Task for BookingTask {
  async fn run(&self, context: Context) -> graph_flow::Result<TaskResult> {
    let request: String = context.get_sync("request").ok_or_else(|| anyhow::anyhow!("'request' 不存在"))?;
    context.set("output", format!("预订处理器处理了请求：\"{}\".结果：模拟预订操作。", request)).await;
    Ok(TaskResult::new(None, NextAction::End))
  }
}

/// 信息处理任务
struct InfoTask;
#[async_trait]
impl Task for InfoTask {
  async fn run(&self, context: Context) -> graph_flow::Result<TaskResult> {
    let request: String = context.get_sync("request").ok_or_else(|| anyhow::anyhow!("'request' 不存在"))?;
    context.set("output", format!("信息处理器处理了请求：\"{}\".结果：模拟信息检索。", request)).await;
    Ok(TaskResult::new(None, NextAction::End))
  }
}

/// 模糊请求处理任务
struct UnclearTask;
#[async_trait]
impl Task for UnclearTask {
  async fn run(&self, context: Context) -> graph_flow::Result<TaskResult> {
    let request: String = context.get_sync("request").ok_or_else(|| anyhow::anyhow!("'request' 不存在"))?;
    context.set("output", format!("模糊处理器处理了请求：\"{}\".结果：无法理解请求。", request)).await;
    Ok(TaskResult::new(None, NextAction::End))
  }
}

/// 决策类型枚举
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, strum::EnumString)]
enum DecisionType {
  Booking, // 预订
  Info,    // 信息
  Unclear, // 模糊
}

/// 路由决策任务 - 根据决策选择下一个任务
struct RouteDecisionTask {
  booking_task_id: String,
  info_task_id: String,
  unclear_task_id: String,
}
impl RouteDecisionTask {
  fn new(booking_task_id: String, info_task_id: String, unclear_task_id: String) -> Self {
    Self { booking_task_id, info_task_id, unclear_task_id }
  }
}
#[async_trait]
impl Task for RouteDecisionTask {
  async fn run(&self, context: Context) -> graph_flow::Result<TaskResult> {
    let decision: DecisionType = context.get_sync("decision").ok_or_else(|| anyhow::anyhow!("决策类型不存在"))?;

    let next_task_id = match decision {
      DecisionType::Booking => &self.booking_task_id,
      DecisionType::Info => &self.info_task_id,
      DecisionType::Unclear => &self.unclear_task_id,
    };

    Ok(TaskResult::new(None, NextAction::GoTo(next_task_id.clone())))
  }
}

/// 路由工作流管理器
struct RouterWorkflow {
  flow_runner: FlowRunner,
  session_storage: Arc<InMemorySessionStorage>,
  coordinator_task: Arc<CoordinatorRouterTask>,
}
impl RouterWorkflow {
  async fn new(system_prompt: &str) -> Result<Self> {
    // 创建任务实例
    let coordinator_task = Arc::new(CoordinatorRouterTask::new(system_prompt)?);
    let booking_task = Arc::new(BookingTask);
    let info_task = Arc::new(InfoTask);
    let unclear_task = Arc::new(UnclearTask);

    // 创建路由决策任务
    let route_decision_task = Arc::new(RouteDecisionTask::new(
      booking_task.id().to_string(),
      info_task.id().to_string(),
      unclear_task.id().to_string(),
    ));

    // 构建图
    let graph = Arc::new(
      GraphBuilder::new("routing_workflow")
        .add_task(coordinator_task.clone())
        .add_task(route_decision_task.clone())
        .add_task(booking_task.clone())
        .add_task(info_task.clone())
        .add_task(unclear_task.clone())
        // 连接任务
        .add_edge(coordinator_task.id(), route_decision_task.id())
        .add_edge(route_decision_task.id(), booking_task.id())
        .add_edge(route_decision_task.id(), info_task.id())
        .add_edge(route_decision_task.id(), unclear_task.id())
        .build(),
    );

    // 创建存储和运行器
    let session_storage = Arc::new(InMemorySessionStorage::new());
    let flow_runner = FlowRunner::new(graph, session_storage.clone());

    Ok(Self { flow_runner, session_storage, coordinator_task })
  }

  /// 执行路由工作流
  async fn execute_workflow(&self, request: String, session_id: &str) -> Result<String> {
    // 创建会话
    let session = Session::new_from_task(session_id.to_string(), self.coordinator_task.id());
    session.context.set("request", request).await;
    (*self.session_storage).save(session).await?;

    // 执行工作流
    let mut final_output = String::new();
    loop {
      let result = self.flow_runner.run(session_id).await?;
      match result.status {
        ExecutionStatus::Completed => {
          // 工作流完成，使用响应作为最终输出
          if let Some(resp) = result.response {
            final_output = resp;
          } else if let Some(session_storage) = self.session_storage.get(session_id).await?
            && let Some(output) = session_storage.context.get("output").await
          {
            final_output = output;
          }
          break;
        }
        ExecutionStatus::Error(err) => return Err(anyhow::anyhow!("Workflow error: {:?}", err)),
        _ => {
          if let Some(resp) = result.response.as_deref() {
            println!("步骤响应: {}", resp);
          }
          continue;
        }
      }
    }

    Ok(final_output)
  }
}

/// 使用如下命令执行程序
/// `cargo run -p adp --example ch02-routing`
#[tokio::main]
async fn main() -> Result<()> {
  // 加载环境变量
  dotenvy::dotenv()?;
  let system_prompt = r#"你是一个智能助手，负责分析用户请求并确定应该将其分配给哪个专业处理程序。
- 如果请求与预订航班或酒店相关，请输出 'Booking'。
- 对于所有其他一般信息问题，请输出 'Info'。
- 如果请求模糊不清或不适合任何类别，请输出 'Unclear'。
仅输出一个单词：'Booking'、'Info'或'Unclear'。"#;

  // 创建路由工作流
  let workflow = RouterWorkflow::new(system_prompt).await?;

  println!("--- 运行**预订**请求 ---");
  let result_a = workflow.execute_workflow("预订一张到重庆的机票。".to_string(), "session_a").await?;
  println!("最终结果 A: {}", result_a);

  println!("\n--- 运行**信息**请求 ---");
  let result_b = workflow.execute_workflow("江苏的省会是哪个城市？".to_string(), "session_b").await?;
  println!("最终结果 B: {}", result_b);

  println!("\n--- 运行**模糊**请求 ---");
  let result_c = workflow.execute_workflow("理讲物子量讲。".to_string(), "session_c").await?;
  println!("最终结果 C: {}", result_c);

  Ok(())
}
```

## 总结

### 核心概念

智能体系统常需应对各种输入和场景，这些都无法通过单一线性流程处理。简单的顺序工作流缺乏基于上下文决策的能力。若缺乏为特定任务选择正确工具或子流程的机制，系统将保持僵化且无法自适应，难以构建处理现实世界复杂多变需求的成熟应用。

### 价值意义

路由模式通过向智能体操作框架引入条件逻辑，提供了标准化解决方案。它使系统能够先分析输入查询以判断其意图或性质，继而动态地将控制流引导至最合适的专用工具、函数或子代理。该决策可通过多种方式驱动，包括大语言模型提示、预定义规则或基于嵌入向量的语义相似度计算。最终，路由将静态的预设执行路径转变为能选择最优操作的灵活且具备上下文感知的工作流。

### 适用原则

当智能体必须根据用户输入或当前状态在多个独立工作流、工具或子代理间做出选择时，应采用路由模式。该模式对需要对接入请求进行分类处理的应用至关重要，例如客户支持机器人需区分销售咨询、技术支持和账户管理问题等场景。

### 核心要点

- 路由使智能体能基于实时条件动态决定工作流中的后续步骤
- 它突破线性执行限制，让智能体可处理多样化输入并自适应调整行为
- 路由逻辑可通过提示工程、规则系统或向量嵌入相似度实现
- 类似 LangGraph 等智能体框架虽然架构各不相同，但都为定义和管理智能体工作流中的路由提供了结构化方案

路由模式是构建真正动态响应式智能体系统的核心环节。通过实施路由机制，我们突破了简单的线性执行流程，使智能体能够智能决策如何处理信息、响应用户输入及调用可用工具或子代理。

从客服聊天机器人到复杂数据处理管道，路由技术的应用场景十分广泛。分析输入并条件式引导工作流的能力，是创建能够应对现实任务多变性的智能体的基础。

LangGraph 的基于图结构提供了可视化且明确的状态与跳转定义，特别适合具有复杂路由逻辑的多步骤工作流；而 rig.rs 和 graph-flow 则在 Rust 生态中提供了更适合 Rust 的解决方案。

掌握路由模式对于构建能够智能应对不同场景的智能体而言至关重要，它能根据上下文提供精准的响应与操作。这是开发多功能、强健型智能体应用的关键组成部分。
