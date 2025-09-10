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
