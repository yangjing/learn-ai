Link Of this Book: <a href="https://nnfs.io/">Book</a>

- Beginner on ML? Check out my Awesome Repository too <a href="https://github.com/Dev-Gaju/Machine-Learning-Starter">Machine Learning StarterMachine Learning Starter</a> <br>
- Langchain, Pinecone and OpenAI with steamlit <a href="https://github.com/Dev-Gaju/-Information-Retrieval-using-Langchain-">A Project</a>
- Learn Python With Data Structure & ALgorithm <a href="https://github.com/Dev-Gaju/Algorithm_and_DataStructer_with_Python"> PYTHON Strater </a>

## Loss

### 负对数似然函数用于计算损失的原因?

负对数似然（Negative Log-Likelihood, NLL）损失函数在分类任务中被广泛使用，特别是在结合 softmax 函数时。以下是其主要原因：

1. 概率解释：
   - Softmax 函数将模型的输出转换为概率分布。
   - 负对数似然损失函数衡量的是模型预测的概率与真实标签之间的差异。
   - 对于正确类别的概率值越接近 1，损失越小；反之，损失越大。
2. 数学性质：
   - 负对数似然损失函数具有良好的数学性质，便于优化。
   - 它是凸函数（对于单个样本），这使得梯度下降等优化算法更容易收敛。
3. 惩罚机制：
   - 当模型对正确类别的预测概率较低时，负对数似然损失会显著增大，从而对错误预测施加更大的惩罚。
   - 这种惩罚机制有助于模型更好地学习和区分不同类别。
4. 数值稳定性：
   - 在实际应用中，直接计算对数似然可能会导致数值不稳定（如概率值非常接近 0 或 1 时）。
   - 使用负对数似然可以避免这些问题，并且在实现中通常结合 log-sum-exp 技巧以提高数值稳定性。
5. 与交叉熵的关系：
   - 负对数似然损失函数实际上等价于交叉熵损失函数（当使用 softmax 作为激活函数时）。
   - 交叉熵损失函数是信息论中的一个概念，用于衡量两个概率分布之间的差异。

#### 总结

负对数似然损失函数通过衡量模型预测的概率与真实标签之间的差异，提供了有效的损失度量方式。它不仅具有良好的数学性质，还能够有效地惩罚错误预测，帮助模型更好地学习和泛化。
