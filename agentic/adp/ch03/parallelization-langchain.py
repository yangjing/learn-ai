import asyncio
from dotenv import load_dotenv

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

load_dotenv()

# 确保已设置您的 API 密钥环境变量（例如：DEEPSEEK_API_KEY）
llm = ChatDeepSeek(model='deepseek-chat', temperature=0.7)

# --- 定义独立链 ---
# 这三个链代表可以并行执行的不同任务。
summarize_chain: Runnable = (
  ChatPromptTemplate.from_messages([('system', '请简洁地总结以下主题：'), ('user', '{topic}')])
  | llm
  | StrOutputParser()
)

questions_chain: Runnable = (
  ChatPromptTemplate.from_messages([('system', '针对以下主题生成三个有趣的问题：'), ('user', '{topic}')])
  | llm
  | StrOutputParser()
)

terms_chain: Runnable = (
  ChatPromptTemplate.from_messages([('system', '从以下主题中识别5-10个关键术语，用逗号分隔：'), ('user', '{topic}')])
  | llm
  | StrOutputParser()
)

# --- 构建并行 + 合成链 ---

# 1. 定义要并行运行的任务块。这些任务的结果，以及原始主题，将被 feed 到下一步。
map_chain = RunnableParallel(
  {
    'summary': summarize_chain,
    'questions': questions_chain,
    'key_terms': terms_chain,
    'topic': RunnablePassthrough(),  # Pass the original topic through
  }
)

# 2. 定义最终合成提示，将并行结果组合起来。
synthesis_prompt = ChatPromptTemplate.from_messages(
  [
    (
      'system',
      """根据以下信息：

    摘要：{summary}

    相关问题：{questions}

    关键术语：{key_terms}

    综合回答。""",
    ),
    ('user', '原始主题：{topic}'),
  ]
)


# 3. 构建完整链，将并行结果直接管道到合成提示，后面是 LLM 和输出解析器。
full_parallel_chain = map_chain | synthesis_prompt | llm | StrOutputParser()


# --- 运行链 ---
async def run_parallel_example(topic: str) -> None:
  """异步调用并行处理链处理指定主题并打印合成结果。

  Args:
    topic: 将被LangChain链处理的输入主题。
  """
  print(f"\n--- 异步调用并行处理链处理主题: '{topic}' ---")

  try:
    # 传递给 `ainvoke` 的输入是单个 'topic' 字符串，
    # 然后传递给 `map_chain` 中的每个可运行组件。
    response = await full_parallel_chain.ainvoke(topic)
    print('\n--- 并行处理链合成结果 ---')
    print(response)
  except Exception as e:
    print(f'\n执行链过程中发生错误：{e}')


# uv run -m agentic.adp.ch03.parallelization-langchain
if __name__ == '__main__':
  test_topic = '空间探索的历史'
  asyncio.run(run_parallel_example(test_topic))
