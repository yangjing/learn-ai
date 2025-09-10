use anyhow::Result;
use async_trait::async_trait;
use graph_flow::{
  Context, ExecutionStatus, FlowRunner, GraphBuilder, InMemorySessionStorage, NextAction, Session, SessionStorage,
  Task, TaskResult,
};
use rig::{
  agent::Agent,
  client::{CompletionClient, ProviderClient},
  completion::Prompt,
  providers::deepseek,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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

    let decision = response.trim();
    context.set("decision", decision).await;

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
