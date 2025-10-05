import torch
import torch.nn.functional as F

def llm_dim_apis_explanation():
    """
    PyTorch 在 LLM 训练推理中常用的涉及 dim 参数的 API。

    这些 API 通过 dim 指定操作沿着哪个维度进行。对于 LLM：
    - 通常 dim=0 是 batch 维度
    - dim=1 是 sequence 长度维度
    - dim=2 是隐藏维度或词汇表维度
    - dim=-1 通常是最后一个维度（方便，不需要指定具体数字）
    """

def demo_llm_dim_apis():
    """演示 LLM 中常用 dim API"""

    # 假设的 LLM tensors
    batch_size = 2
    seq_len = 3
    vocab_size = 5
    hidden_dim = 4

    # 注意力分数矩阵: shape (batch, seq_len, seq_len)
    attn_scores = torch.randn(batch_size, seq_len, seq_len)
    print(f"注意力分数 tensor shape: {attn_scores.shape}")

    # 分类 logits: shape (batch, seq_len, vocab)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    print(f"分类 logits tensor shape: {logits.shape}")

    # 隐藏状态: shape (batch, seq_len, hidden)
    hidden = torch.randn(batch_size, seq_len, hidden_dim)
    print(f"隐藏状态 tensor shape: {hidden.shape}")

    print("\n" + "="*50)
    print("常用 LLM API 解析")
    print("="*50)

    # 1. torch.softmax(dim=-1) - 在词汇表维度上做 softmax
    print("\n1. torch.softmax(dim=-1)")
    print("在 LLM 中：将 raw logits 转换为概率分布用于采样或计算损失")
    print(f"logits shape: {logits.shape}")
    probs = F.softmax(logits, dim=-1)
    print(f"softmax(logits, dim=-1) shape: {probs.shape}")
    print("解释: 对每个样本、每个位置的所有词汇表 logit 计算 softmax，概率和=1")
    # 验证概率和
    prob_sum = probs.sum(dim=-1)
    print(f"每位置概率和 (应接近1): \n{prob_sum}")

    # 2. torch.argmax(dim=-1) - 在词汇表维度上找最大概率单词
    print("\n2. torch.argmax(dim=-1)")
    print("在 LLM 中：greedy decoding，选最可能的单词")
    pred_tokens = torch.argmax(logits, dim=-1)
    print(f"argmax(logits, dim=-1): {pred_tokens}")
    print("逻辑: 对每个 batch、每个 position，选 vocab 维度中最大 logit 的索引")

    # 3. torch.sum(dim=1) - 沿 sequence 维度求和
    print("\n3. torch.sum(dim=1)")
    print("在 LLM 中：用于 cross entropy loss 计算前对语义长度的 log prob 求和")
    log_probs = F.log_softmax(logits, dim=-1)
    seq_log_probs = log_probs.sum(dim=1)  # 对整个序列的 log prob 求和
    print(f"log probs shape: {log_probs.shape}")
    print(f"sum(dim=1) shape: {seq_log_probs.shape}")
    print("解释: dim=1 是 sequence 位置，每个样本在长度维度 ('整个序列') 上求和")

    # 4. torch.mean(dim=1) - 沿 sequence 维度取平均
    print("\n4. torch.mean(dim=1)")
    print("在 LLM 中：mean pooling over sequence for sentence embedding 或 decoder start token")
    pooled = hidden.mean(dim=1)  # 对序列位置取平均，得到句子 embedding
    print(f"hidden shape: {hidden.shape}")
    print(f"mean(dim=1) shape: {pooled.shape}")
    print("解释: 将变长序列 pooling 成固定大小 embedding，dim=1 是时间维度")

    # 5. torch.topk(dim=-1) - 在 vocab 维度找 top-k
    print("\n5. torch.topk(dim=-1, k=3)")
    print("在 LLM 中：top-k sampling 或 beam search")
    values, indices = torch.topk(logits, k=3, dim=-1)
    print("应用场景: 限制采样到最可能的 k 个 token 中")
    print(f"top 3 values shape: {values.shape}")
    print(f"top 3 indices shape: {indices.shape}")
    print("逻辑: 在 vocab 维度找到最大的 k 个值和对应索引")

    # 6. torch.squeeze(dim=1) / torch.unsqueeze(dim=1)
    print("\n6. torch.squeeze 或 torch.unsqueeze (dim)")
    print("在 LLM 中：调整张量形状，用于广播或适配不同操作")
    # 假设 decoder hidden 需要广播给所有 vocab positions
    decoder_hidden = pooled.unsqueeze(dim=1)  # (batch, 1, hidden) for broadcasting
    print(f"pooled shape: {pooled.shape}")
    print(f"unsqueeze(dim=1) shape: {decoder_hidden.shape}")
    print("解释: 在 dim=1 位置插入维度，用于后续 broadcasting 与 vocab logits 对齐")

    # 7. torch.cat(dim=1) - 在 sequence 维度拼接
    print("\n7. torch.cat(dim=1)")
    print("在 LLM 中：拼接输入序列和目标序列用于训练")
    prefix = hidden[:, :1]  # 取第一个 token 作为 prefix
    suffix = hidden[:, 1:]  # 取剩余序列
    concatenated = torch.cat([prefix, suffix], dim=1)
    print(f"prefix shape: {prefix.shape}, suffix shape: {suffix.shape}")
    print(f"cat([prefix, suffix], dim=1) shape: {concatenated.shape}")
    print("逻辑: 在 sequence 维度 (dim=1) 上拼接 tensor")

    # 8. torch.einsum - 任意维度缩并 (非常灵活)
    print("\n8. torch.einsum - 任意维度操作")
    print("在 LLM 中：用于复杂的多头注意力计算或张量积")
    # 简化版 attention: (batch, seq, hidden) @ (batch, hidden, vocab) -> (batch, seq, vocab)
    weight = torch.randn(hidden_dim, vocab_size)  # 投影矩阵
    output = torch.einsum('bsh,hv->bsv', hidden, weight)
    print("einsum('bsh,hv->bsv', hidden, weight)")
    print(f"hidden shape: {hidden.shape}, weight shape: {(hidden_dim, vocab_size)}")
    print(f"output shape: {output.shape}")
    print("解释: b=batch, s=seq, h=hidden, v=vocab。点积 batch 和 seq 所有位置到所有 vocab")

    print("\n" + "="*50)
    print("总结：")
    print("- dim=-1: 通常操作最后的维度 (vocab/feature)")
    print("- dim=0: batch 维度，不常用 (影响多个样本)")
    print("- dim=1: sequence 维度，语言模型的核心操作")
    print("- dim=2: 可能表示多个头或通道")
    print("理解关键: LLM 处理序列数据，dim 控制是在 batch、序列、特征哪个层面操作")

if __name__ == "__main__":
    demo_llm_dim_apis()
