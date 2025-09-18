# Agentic 设计模式 - 反思模式

## 反思模式（Reflection Pattern）

在前文中，我们已经探讨了智能体的几种基础运作模式：实现顺序执行的链式模式、支持动态路径选择的路由模式，以及用于并发任务执行的并行模式。这些模式使智能体能够更高效灵活地处理复杂任务。然而，即便采用精密的工作流程，智能体的初始输出或计划仍可能不够完善、准确或全面。此时便需要引入反思模式（Reflection Pattern）。

反思模式是指智能体对自身的工作成果、输出内容或内部状态进行评估，并借助评估结果提升表现或优化反馈的一种机制。这是一种自我修正与改进的过程，使智能体能够基于反馈、内部审查或与预期标准的对比，迭代完善输出结果或调整执行策略。在某些情况下，也可引入专职分析初始智能体输出的辅助智能体来协助完成反思。

与直接传递输出至下一环节的简单链式模式，或仅进行路径选择的路由模式不同，反思模式构建了一种反馈循环机制。智能体不仅生成输出，还会对该输出（及其生成过程）进行审查，识别潜在问题与改进空间，并运用这些洞察生成优化版本或调整后续行动。

该流程通常包含以下环节：

1. **执行阶段**：智能体执行任务或生成初始输出；
2. **评估/审查阶段**：智能体（通常通过另一次 LLM 调用或规则集）对前序结果进行分析，评估内容可涉及事实准确性、逻辑连贯性、风格一致性、完成度、指令遵循度等维度；
3. **反思/优化阶段**：基于评估结论，智能体制定改进方案，可能包括生成优化输出、调整后续参数甚至修订整体计划；
4. **迭代循环（常见但非必选）**：优化后的输出或调整后的策略将进入新一轮执行，反思流程可循环直至达成满意结果或满足终止条件。

反思模式的一个关键高效实现方案，是将流程分离为两个独立逻辑角色：执行者（Producer）与评审者（Critic），这也常被称为"生成者-评审者"模型。虽然单个智能体可完成自我反思，但采用两个专职智能体（或通过不同系统指令分离的 LLM 调用）通常能产生更可靠且无偏差的结果。

1. **执行者智能体**：主要负责任务的初始执行，专注于内容生成（如编写代码、起草文稿、制定计划），根据初始指令产生第一版输出；
2. **评审者智能体**：专司评估执行者的输出成果，其被赋予不同的指令集与角色定位（如"资深软件工程师""严谨的事实核查员"），依据特定标准（事实准确性、代码质量、文体要求、完成度等）进行分析，致力于发现缺陷、提出改进建议并提供结构化反馈。
   这种职责分离机制能有效避免智能体自查工作时的"认知偏差"。评审者以全新视角审视输出，专注挖掘错误与改进空间。其反馈将返回至执行者，引导生成优化版本。

实施反思模式通常需要构建含反馈循环的工作流，可通过代码中的迭代循环或支持状态管理与条件转换的框架实现。虽然单次评估优化可在 LangChain/LangGraph、ADK 或 Crew.AI 链中完成，但真正的迭代反思往往需要更复杂的协调机制。

反思模式对于构建能产出高质量结果、处理复杂任务且具备一定自我认知与适应能力的智能体至关重要。它推动智能体从单纯执行指令向更高级的问题解决与内容生成模式演进。

值得注意的是，反思模式与目标设定及监控存在深度关联：目标为智能体自我评估提供终极基准，监控则追踪进展轨迹。在实际应用中，反思常作为修正引擎，借助监控反馈分析偏差并调整策略。这种协同效应将智能体从被动执行者转化为具有目标导向的适应性系统。
此外，当 LLM 具备对话记忆能力时，反思模式的有效性将显著提升。对话历史为评估阶段提供关键上下文，使智能体能结合先前交互、用户反馈与动态目标进行综合研判，从而从历史批评中学习并避免重复错误。若无记忆功能，每次反思仅是孤立事件；具备记忆后，反思即成为持续累积的过程，每个循环都基于前次成果推进，最终实现更智能且贴合语境的优化。

## 实际应用与用例

反思模式在输出质量、准确性或复杂约束遵循度至关重要的场景中具有重要价值：

1. **创意写作与内容生成**：优化生成的文本、故事、诗歌或营销文案。
   - 应用场景：智能体撰写博客文章。
     - 反思流程：生成初稿后，从行文流畅度、语气和清晰度等角度进行评审，并基于反馈重写内容，循环该过程直至文章符合质量标准。
     - 优势：产出更精炼、效果更突出的内容。
2. **代码生成与调试**：编写代码，识别错误并修复问题。
   - 应用场景：智能体编写 Python 函数。
     - 反思流程：首先生成代码，运行测试或静态分析，找出错误或低效之处，随后根据发现的问题修改代码。
     - 优势：生成更健壮、功能性更强的代码。
3. **复杂问题求解**：在多步推理任务中评估中间步骤或建议方案。
   - 应用场景：智能体解决逻辑谜题。
     - 反思流程：提出一个步骤，评估其是否推进了解题进程或引发矛盾，必要时回退或尝试其他步骤。
     - 优势：提升智能体在复杂问题空间中的导航能力。
4. **摘要与信息整合**：优化摘要的准确性、完整性和简洁性。
   - 应用场景：智能体总结长文档。
     - 反思流程：生成初步摘要，对照原文关键点查漏补缺，通过迭代优化提高摘要的准确性和完整性。
     - 优势：生成更准确、全面的摘要。
5. **规划与策略制定**：评估拟议计划，识别潜在缺陷或改进点。
   - 应用场景：智能体制定实现目标的行动序列。
     - 反思流程：生成计划，模拟执行或根据约束评估可行性，依据评估结果修订计划。
     - 优势：形成更有效、更符合实际的计划。
6. **对话系统**：回顾对话历史以保持上下文连贯、纠正误解或提升回复质量。
   - 应用场景：客服聊天机器人。
     - 反思流程：在用户回复后，重新审视对话历史及最新生成的消息，确保回应连贯且精准回应用户最新输入。
     - 优势：实现更自然、高效的人机对话。

反思模式为智能体系统赋予了“元认知”能力，使其能够从自身输出与处理过程中持续学习，进而产生更智能、更可靠、更高质量的成果。

## 代码示例

这个代码实现了基于 LangGraph 的反思模式（Reflection Pattern），通过两个智能体的协作来逐步改进代码质量：

- 生成器智能体：负责生成或改进 Python 代码
- 反思器智能体：作为资深代码审查员，评估代码质量并提供改进建议
- 迭代优化：通过多轮反思循环，持续改进代码直到满足要求

```python
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
NodeResult = dict[str, Any]
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
```

### 关键技术点

1. LangGraph 最佳实践

- 使用 `StateGraph` 构建状态机
- 采用 `add_messages` 注解管理消息历史
- 实现条件边缘控制流程分支

2. 状态更新策略：遵循 LangGraph 最佳实践，返回新的状态字典而非修改原状态：

```python
return {
  'messages': [response],
  'current_code': current_code,
  'iteration': state['iteration'] + 1
}
```

3. 智能停止机制

- 质量达标：反思器评估为 `CODE_IS_PERFECT`
- 迭代上限：达到最大迭代次数（默认 3 次）
- 条件路由：使用模式匹配优雅处理多种停止条件

## 要点速览

**核心内容：** 智能体的初始输出往往不够完善，常存在准确性不足、完整性欠缺或难以满足复杂要求等问题。基础智能体工作流缺乏内置机制，无法自主识别并修正错误。为解决这一局限，反思模式要求智能体对自身工作进行评估；更可靠的做法是引入一个独立的逻辑智能体作为评审者，从而避免无论质量如何都将初始响应作为最终输出。

**价值所在：** 反思模式通过引入自我修正与优化的机制，构建出一个反馈循环：先由“执行者”智能体生成输出，再由“评审者”智能体（或执行者自身）依据预设标准进行评估，并基于评估结果生成改进版本。这种“生成-评估-优化”的迭代过程持续提升最终结果的质量，使其更准确、连贯和可靠。

**适用原则：** 当最终输出的质量、准确性和细节丰富度比处理速度和成本更为重要时，即应使用反思模式。该模式特别适用于生成精炼的长篇内容、编写与调试代码、制定详细计划等任务。若任务要求较高的客观性或需要执行者可能忽略的专业评审，则应引入独立的评审者智能体。

## 核心要点

- 反思模式的核心优势在于能够通过迭代实现自我修正与输出优化，从而显著提升结果的质量、准确性及对复杂指令的遵循程度。
- 该模式依托 **“执行—评估/评审—优化”** 构成的反馈循环，尤其适用于对输出质量、准确性或细节要求较高的任务。
- 一种高效的实现方式是采用“生产者-评审者”模型，通过专职智能体（或特定角色提示）对初始输出进行评价。这种角色分离既增强了客观性，也有助于提供更专业化、结构化的反馈。
- 然而，该模式也会带来延迟增加、计算成本上升、更容易超出模型上下文长度限制或被 API 服务限流等问题。
- 这一模式使得智能体能够在不断迭代中自我改进，逐步提升性能表现。

## 总结

反思模式为智能体工作流提供了一种关键的自我修正机制，使其能够突破单次执行的局限，通过迭代持续改进输出结果。该模式构建了一个循环流程：系统生成输出后，依据既定标准对其评估，并借助评估信息生成优化结果。评估可由智能体自身完成（自我反思），但通常更有效的做法是引入一个独立的评审者智能体——这也是该模式中一项重要的架构决策。

尽管完全自主的多步反思过程需要较强的状态管理架构支持，但其核心原理在单一的“生成–评审–优化”循环中已得到充分体现。作为一种控制结构，反思模式可与其他基础模式结合使用，共同构建更稳健、功能更复杂的智能体系统。
