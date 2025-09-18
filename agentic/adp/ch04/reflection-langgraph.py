"""
LangGraph 实现的反思模式 (Reflection Pattern)

这个模块演示了如何使用 LangGraph 构建一个反思循环，
通过生成器和反思器两个智能体协作来逐步改进代码质量。
"""

from typing import TypedDict, Literal, Annotated, Any, Dict
from dotenv import load_dotenv

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# 加载环境变量
load_dotenv()


# 定义状态类型
class ReflectionState(TypedDict):
  """反思循环的状态定义"""

  messages: Annotated[list[BaseMessage], add_messages]
  current_code: str
  critique: str
  iteration: int
  max_iterations: int
  task_prompt: str
  is_perfect: bool


# 类型别名
NodeResult = Dict[str, Any]
ConditionalEdge = Literal['continue', 'end']


class ReflectionAgent:
  """反思智能体类，封装生成器和反思器的逻辑"""

  def __init__(self, model: str = 'deepseek-chat', temperature: float = 0.1):
    """初始化反思智能体

    Args:
        model: 使用的模型名称
        temperature: 模型温度参数
    """
    self.llm = ChatDeepSeek(model=model, temperature=temperature)
    self.task_prompt = """
        你的任务是创建一个名为 `calculate_factorial` 的 Python 函数。这个函数需要满足以下要求：

        1. 接受一个整数 `n` 作为输入
        2. 计算其阶乘 (n!)
        3. 包含清晰的文档字符串说明函数功能
        4. 处理边界情况：0 的阶乘是 1
        5. 处理无效输入：如果输入是负数，抛出 ValueError
        """

  def generator_node(self, state: ReflectionState) -> NodeResult:
    """代码生成器节点

    根据当前状态生成或改进代码
    """
    print(f'\n{"=" * 25} 反思循环：第 {state["iteration"]} 次迭代 {"=" * 25}')

    if state['iteration'] == 1:
      print('\n>>> 阶段 1：生成初始代码...')
      # 第一次迭代：生成初始代码
      messages = [HumanMessage(content=self.task_prompt)]
    else:
      print('\n>>> 阶段 1：根据反馈改进代码...')
      # 后续迭代：基于反馈改进代码
      messages = state['messages'] + [HumanMessage(content=f'请根据以下反馈改进代码：\n{state["critique"]}')]

    response = self.llm.invoke(messages)
    current_code = response.content

    print(f'\n--- 生成的代码 (v{state["iteration"]}) ---')
    print(current_code)

    return {'messages': [response], 'current_code': current_code, 'iteration': state['iteration'] + 1}

  def reflector_node(self, state: ReflectionState) -> NodeResult:
    """代码反思器节点

    评估当前代码质量并提供改进建议
    """
    print('\n>>> 阶段 2：反思生成的代码...')

    # 构建反思提示
    reflector_messages = [
      SystemMessage(
        content="""
            你是一位资深软件工程师和 Python 专家。
            你的职责是进行细致的代码审查。
            请根据原始任务要求，严格评估提供的 Python 代码。
            查找错误、代码风格问题、遗漏的边界情况和改进空间。
            如果代码完美并满足所有要求，请回复单独的短语 'CODE_IS_PERFECT'。
            否则，请提供带编号的改进建议列表。
            """
      ),
      HumanMessage(
        content=f"""
            原始任务：
            {self.task_prompt}

            待审查的代码：
            {state['current_code']}
            """
      ),
    ]

    response = self.llm.invoke(reflector_messages)
    critique = response.content

    # 检查是否完美
    is_perfect = 'CODE_IS_PERFECT' in critique

    if is_perfect:
      print('\n--- 反馈 ---')
      print('未发现进一步的改进建议。代码质量令人满意。')
    else:
      print('\n--- 反馈 ---')
      print(critique)

    return {'critique': critique, 'is_perfect': is_perfect, 'messages': [response]}

  def should_continue(self, state: ReflectionState) -> ConditionalEdge:
    """条件边缘：决定是否继续迭代

    Returns:
        "continue": 继续迭代
        "end": 结束循环
    """
    # 检查停止条件
    match (state['is_perfect'], state['iteration'] >= state['max_iterations']):
      case (True, _):
        return 'end'
      case (_, True):
        print(f'\n达到最大迭代次数 {state["max_iterations"]}，停止循环。')
        return 'end'
      case _:
        return 'continue'

  def create_graph(self):
    """创建并配置 LangGraph 状态图"""
    # 创建状态图
    workflow = StateGraph(ReflectionState)

    # 添加节点
    workflow.add_node('generator', self.generator_node)
    workflow.add_node('reflector', self.reflector_node)

    # 设置入口点
    workflow.add_edge(START, 'generator')

    # 添加连接
    workflow.add_edge('generator', 'reflector')

    # 添加条件连接
    workflow.add_conditional_edges('reflector', self.should_continue, {'continue': 'generator', 'end': END})

    return workflow.compile()

  def run_reflection_loop(self, max_iterations: int = 3) -> str:
    """运行完整的反思循环

    Args:
        max_iterations: 最大迭代次数

    Returns:
        最终优化的代码
    """
    # 创建图
    app = self.create_graph()

    # 初始状态
    initial_state: ReflectionState = {
      'messages': [],
      'current_code': '',
      'critique': '',
      'iteration': 1,
      'max_iterations': max_iterations,
      'task_prompt': self.task_prompt,
      'is_perfect': False,
    }

    # 运行图
    final_state = app.invoke(initial_state)

    # 输出最终结果
    print('\n' + '=' * 30 + ' 最终结果 ' + '=' * 30)
    print('\n反思过程完成后的最终优化代码：\n')
    print(final_state['current_code'])

    return final_state['current_code']


def main():
  """主函数：演示反思模式的使用"""
  try:
    # 创建反思智能体
    agent = ReflectionAgent()
    # 运行反思循环
    agent.run_reflection_loop(max_iterations=3)
    print('\n✅ 反思循环成功完成')
  except Exception as e:
    print(f'\n❌ 运行过程中出现错误：{e}')
    raise


# uv run -m agentic.adp.ch04.reflection-langgraph
if __name__ == '__main__':
  main()
