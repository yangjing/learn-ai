import os
from dotenv import load_dotenv

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage


# Load environment variables from .env file (for DEEPSEEK_API_KEY)
load_dotenv()


# Initialize the Chat LLM. We use deepseek-chat for better reasoning.
# A lower temperature is used for more deterministic outputs.
llm = ChatDeepSeek(model='deepseek-chat', temperature=0.1)


def run_reflection_loop():
  """Demonstrates a multi-step AI reflection loop to progressively improve a Python function."""

  # --- The Core Task ---
  task_prompt = """
   Your task is to create a Python function named `calculate_factorial`. This function should do the following:

   1.  Accept a single integer `n` as input.
   2.  Calculate its factorial (n!).
   3.  Include a clear docstring explaining what the function does.
   4.  Handle edge cases: The factorial of 0 is 1.
   5.  Handle invalid input: Raise a ValueError if the input is a negative number.
   """

  # --- The Reflection Loop ---
  max_iterations = 3
  current_code = ''

  # We will build a conversation history to provide context in each step.
  message_history = [HumanMessage(content=task_prompt)]

  for i in range(max_iterations):
    print('\n' + '=' * 25 + f' REFLECTION LOOP: ITERATION {i + 1} ' + '=' * 25)

    # --- 1. GENERATE / REFINE STAGE ---
    # In the first iteration, it generates. In subsequent iterations, it refines.
    if i == 0:
      print('\n>>> STAGE 1: GENERATING initial code...')
      # The first message is just the task prompt.
      response = llm.invoke(message_history)
      current_code = response.content
    else:
      print('\n>>> STAGE 1: REFINING code based on previous critique...')
      # The message history now contains the task,
      # the last code, and the last critique.
      # We instruct the model to apply the critiques.
      message_history.append(HumanMessage(content='Please refine the code using the critiques provided.'))
      response = llm.invoke(message_history)
      current_code = response.content

    print('\n--- Generated Code (v' + str(i + 1) + ') ---\n' + current_code)
    message_history.append(response)  # Add the generated code to history

    # --- 2. REFLECT STAGE ---
    print('\n>>> STAGE 2: REFLECTING on the generated code...')

    # Create a specific prompt for the reflector agent.
    # This asks the model to act as a senior code reviewer.
    reflector_prompt = [
      SystemMessage(
        content="""
               You are a senior software engineer and an expert in Python.
               Your role is to perform a meticulous code review.
               Critically evaluate the provided Python code based
               on the original task requirements.
               Look for bugs, style issues, missing edge cases,
               and areas for improvement.
               If the code is perfect and meets all requirements,
               respond with the single phrase 'CODE_IS_PERFECT'.
               Otherwise, provide a bulleted list of your critiques.
               """
      ),
      HumanMessage(content=f'Original Task:\n{task_prompt}\n\nCode to Review:\n{current_code}'),
    ]

    critique_response = llm.invoke(reflector_prompt)
    critique = critique_response.content

    # --- 3. STOPPING CONDITION ---
    if 'CODE_IS_PERFECT' in critique:
      print('\n--- Critique ---\nNo further critiques found. The code is satisfactory.')
      break

    print('\n--- Critique ---\n' + critique)

    # Add the critique to the history for the next refinement loop.
    message_history.append(HumanMessage(content=f'Critique of the previous code:\n{critique}'))

  print('\n' + '=' * 30 + ' FINAL RESULT ' + '=' * 30)
  print('\nFinal refined code after the reflection process:\n')
  print(current_code)


# uv run -m agentic.adp.ch04.reflection-langchain
if __name__ == '__main__':
  run_reflection_loop()
