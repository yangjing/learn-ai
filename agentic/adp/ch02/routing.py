from dotenv import load_dotenv

from langchain_deepseek import ChatDeepSeek
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableSerializable

load_dotenv()


def booking_handler(request: str) -> str:
  """Simulates the Booking Agent handling a request"""
  print('\n--- DELEGATING TO BOOKING HANDLER ---')
  return f'预订处理器处理了请求："{request}"。结果：模拟预订操作。'


def info_handler(request: str) -> str:
  """Simulates the Info Agent handling a request"""
  print('\n--- DELEGATING TO INFO HANDLER ---')
  return f'信息处理器处理了请求："{request}"。结果：模拟信息检索。'


def unclear_handler(request: str) -> str:
  """Handles requests that couldn't be delegated."""
  print('\n--- HANDLING UNCLEAR REQUEST ---')
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

  # Create the RunnableBranch. It takes the output of the router chain
  delegation_branch = RunnableBranch(
    (lambda x: x['decision'].strip() == '预订', branches['预订']),
    (lambda x: x['decision'].strip() == '信息', branches['信息']),
    branches['模糊'],
  )

  # Combine the router chain and the delegation branch into a single runnable.
  # The router chain's output ('decision') is passed along with the original input ('request') to the delegation_branch
  coordinator_agent = (
    {
      'decision': coordinator_router_chain,
      'request': RunnablePassthrough(),
    }
    | delegation_branch
    | (lambda x: x['output'])
  )

  return coordinator_agent


def main():
  """路由机制
  - Routing enables agents to make dynamic decisions about the next step in a workflow based on conditions.
  - It allows agents to handle diverse inputs and adapt their behavior, moving beyond linear execution.
  - Routing logic can be implemented using LLMs, rule-based systems, or embedding similarity.
  - Frameworks like LangGraph and Google ADK provide structured ways to define and manage routing within
    agent workflows, albeit with different architectural approaches.
  """
  llm = ChatDeepSeek(model='deepseek-chat', temperature=0)

  coordinator_agent = build(llm)

  print('--- Running with a booking request ---')
  request_a = '预订一张到重庆的机票。'
  result_a = coordinator_agent.invoke({'request': request_a})
  print(f'最终结果 A: {result_a}')

  print('\n--- Running with an info request ---')
  request_b = '重庆的天气怎么样？'
  result_b = coordinator_agent.invoke({'request': request_b})
  print(f'最终结果 B: {result_b}')

  print('\n--- Running with an unclear request ---')
  request_c = '理讲物子量讲。'
  result_c = coordinator_agent.invoke({'request': request_c})
  print(f'最终结果 C: {result_c}')


# uv run -m agentic.adp.02-routing
if __name__ == '__main__':
  main()
