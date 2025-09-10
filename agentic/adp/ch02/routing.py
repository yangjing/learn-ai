from typing import Any
from dotenv import load_dotenv

from langchain_deepseek import ChatDeepSeek
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableSerializable

load_dotenv()


def booking_handler(request: str) -> str:
  print('\n--- 委托给预订处理器 ---')
  return f'预订处理器处理了请求："{request}"。结果：模拟预订操作。'


def info_handler(request: str) -> str:
  print('\n--- 委托给信息处理器 ---')
  return f'信息处理器处理了请求："{request}"。结果：模拟信息检索。'


def unclear_handler(request: str) -> str:
  print('\n--- 处理模糊请求 ---')
  return f'协调器无法委托请求："{request}"。请澄清。'


def build(llm: BaseChatModel) -> RunnableSerializable:
  # Define Coordinator Router Chain. This chain decides which handler to delegate to.
  coordinator_router_prompt = ChatPromptTemplate.from_messages(
    [
      {
        'role': 'system',
        'content': """你是一个智能助手，负责分析用户请求并确定应该将其分配给哪个专业处理程序。
    - 如果请求与预订航班或酒店相关，请输出 '预订'。
    - 对于所有其他一般信息问题，请输出 '信息'。
    - 如果请求模糊不清或不适合任何类别，请输出 '模糊'。
    仅输出一个单词：'预计'、'信息'或'模糊'。""",
      },
      {'role': 'user', 'content': '{request}'},
    ]
  )

  coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()

  # Define the Delegation Logic. Use RunnableBranch to route based on the router chain's output.
  branches = {
    '预订': RunnablePassthrough.assign(output=lambda x: booking_handler(x['request']['request'])),
    '信息': RunnablePassthrough.assign(output=lambda x: info_handler(x['request']['request'])),
    '模糊': RunnablePassthrough.assign(output=lambda x: unclear_handler(x['request']['request'])),
  }

  # Define condition functions with proper type annotations
  def is_booking_request(x: dict[str, Any]) -> bool:
    return x['decision'].strip() == '预订'

  def is_info_request(x: dict[str, Any]) -> bool:
    return x['decision'].strip() == '信息'

  # Create the RunnableBranch. It takes the output of the router chain
  delegation_branch = RunnableBranch(
    (is_booking_request, branches['预订']),
    (is_info_request, branches['信息']),
    branches['模糊'],
  )

  # Combine the router chain and the delegation branch into a single runnable.
  # The router chain's output ('decision') is passed along with the original input ('request') to the delegation_branch
  coordinator_agent = (
    {'decision': coordinator_router_chain, 'request': RunnablePassthrough()}
    | delegation_branch
    | (lambda x: x['output'])
  )

  return coordinator_agent


def main():
  """路由机制"""
  llm = ChatDeepSeek(model='deepseek-chat', temperature=0)

  coordinator_agent = build(llm)

  print('--- 运行**预订**请求 ---')
  request_a = '预订一张到重庆的机票。'
  result_a = coordinator_agent.invoke({'request': request_a})
  print(f'最终结果 A: {result_a}')

  print('\n--- 运行**信息**请求 ---')
  request_b = '江苏的省会是哪个城市？'
  result_b = coordinator_agent.invoke({'request': request_b})
  print(f'最终结果 B: {result_b}')

  print('\n--- 运行**模糊**请求 ---')
  request_c = '理讲物子量讲。'
  result_c = coordinator_agent.invoke({'request': request_c})
  print(f'最终结果 C: {result_c}')


# uv run -m agentic.adp.ch02.routing
if __name__ == '__main__':
  main()
