import os
from typing import Annotated
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

load_dotenv()


class State(TypedDict):
  messages: Annotated[list, add_messages]
  """messages: 对话历史，使用 add_messages 注解自动合并消息f"""

  name: str
  """存储查询的人名"""

  birthday: str
  """存储查询的生日"""


# InjectedToolCallId: 自动注入工具调用ID，用于消息关联。可以防止信息泄露给 LLM
@tool
def human_assistance(name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
  """人机交互工具，请求人类协助，确认查询的生日是否正确。"""

  # 中断，等待人类协助
  human_response = interrupt(
    {
      'question': '这是正确的吗？',
      'name': name,
      'birthday': birthday,
    },
  )
  if human_response.get('正确', '').lower().startswith('是'):
    verified_name = name
    verified_birthday = birthday
    response = '正确'
  else:
    verified_name = human_response.get('name', name)
    verified_birthday = human_response.get('birthday', birthday)
    response = f'修正: {human_response}'

  state_update = {
    'name': verified_name,
    'birthday': verified_birthday,
    'messages': [ToolMessage(response, tool_call_id=tool_call_id)],
  }
  return Command(update=state_update)


# from langchain.chat_models import init_chat_model

# llm = init_chat_model('deepseek:deepseek-chat')
llm = ChatOpenAI(
  model='Qwen/Qwen3-30B-A3B-Instruct-2507',
  openai_api_key=os.environ['SILICONFLOW_API_KEY'],
  openai_api_base='https://api.siliconflow.cn/v1',
  streaming=True,
)
search_tool = TavilySearch(max_results=2)
tools = [search_tool, human_assistance]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
  """聊天机器人节点"""

  message = llm_with_tools.invoke(state['messages'])
  # 移除断言，允许多个工具调用
  # assert len(message.tool_calls) <= 1
  return {'messages': [message]}


def build_graph():
  graph_builder = StateGraph(State)
  graph_builder.add_node('chatbot', chatbot)

  tool_node = ToolNode(tools=tools)
  graph_builder.add_node('tools', tool_node)

  graph_builder.add_conditional_edges(
    'chatbot',
    tools_condition,
  )
  graph_builder.add_edge('tools', 'chatbot')
  graph_builder.add_edge(START, 'chatbot')
  graph_builder.add_edge('chatbot', END)

  memory = InMemorySaver()
  graph = graph_builder.compile(checkpointer=memory)
  return graph


# uv run -m langgraph.tutorials.tutorial-custom-state
if __name__ == '__main__':
  graph = build_graph()
  config = {'configurable': {'thread_id': '1'}}

  # 执行步骤：
  # 1. 用户输入查询
  # 2. chatbot 节点调用LLM
  # 3. LLM决定使用 tavily_search 工具
  # 4. tools 节点执行搜索
  # 5. 返回搜索结果给 chatbot
  # 6. LLM分析结果并调用 human_assistance 工具
  # 7. human_assistance 中断执行，等待人工确认
  print('# 1. 用户查询。提示 chatbot')
  user_input = '你能查一下羊八井出生的时间吗？当你得到答案后，使用 human_assistance 工具进行审查。'

  for event in graph.stream(
    {'messages': [{'role': 'user', 'content': user_input}]},
    config,
    stream_mode='values',
  ):
    if 'messages' in event:
      event['messages'][-1].pretty_print()

  # 执行步骤：
  # 1. 人工提供正确的信息
  # 2. 使用 Command(resume=...) 恢复执行
  # 3. 更新状态中的 name 和 birthday
  # 4. 继续执行流程
  print('# 2. 人工干预。聊天机器人未能识别正确的日期，请提供以下信息：')
  human_command = Command(
    resume={
      'name': '羊八井',
      'birthday': '1985-10-23',
    },
  )
  for event in graph.stream(human_command, config, stream_mode='values'):
    if 'messages' in event:
      event['messages'][-1].pretty_print()
