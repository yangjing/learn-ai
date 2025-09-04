from dotenv import load_dotenv

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def main():
  """多步骤提示链示例

  - 提示链将复杂任务分解为一系列较小的、重点明确的步骤。这有时也被称为管道模式。
  - 链中的每个步骤都涉及一个LLM调用或处理逻辑，使用前一步骤的输出作为输入。
  - 这种模式提高了与语言模型进行复杂交互的可靠性和可管理性。
  - 像LangChain/LangGraph和Google ADK这样的框架提供了强大的工具来定义、管理和执行这些多步骤序列。
  """
  # 初始化 llm
  llm = ChatDeepSeek(model='deepseek-chat', temperature=0)

  # Prompt 1: 抽取信息
  prompt_extract = ChatPromptTemplate.from_template('从以下文本中提取技术规格：\n\n{text_input}')

  # Prompt 2: 转换为 JSON
  prompt_transform = ChatPromptTemplate.from_template(
    '将以下规格信息转换为包含"cpu"、"memory"和"storage"键的JSON对象：\n\n{specifications}'
  )

  # 使用 LCEL 构建链
  extraction_chain = prompt_extract | llm | StrOutputParser()
  full_chain = {'specifications': extraction_chain} | prompt_transform | llm | StrOutputParser()

  input_text = '这款新笔记本电脑配备了3.5GHz的8核处理器、16GB内存和1TB NVMe固态硬盘。'

  # 使用输入文本执行链
  final_result = full_chain.invoke({'text_input': input_text})

  print('\n--- 最终 JSON 输出 ---')
  print(final_result)


# 可以直接运行该脚本，使用如下命令
# uv run -m agentic.adp.ch01.multi_step
if __name__ == '__main__':
  main()
