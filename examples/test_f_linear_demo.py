import torch
import torch.nn.functional as F

def demo_f_linear():
    print("=== F.linear API Demo ===\n")

    # F.linear 的语法：torch.nn.functional.linear(input, weight, bias=None)
    # 作用：执行线性变换 y = input @ weight.T + bias
    # 其中：
    # - input: 输入张量，形状为 (*, in_features)
    # - weight: 权重矩阵，形状为 (out_features, in_features)
    # - bias: 可选偏置向量，形状为 (out_features,)

    # 示例 1: 基本用法，无偏置
    print("示例 1: 基本用法，无偏置")
    input_tensor = torch.randn(2, 3)  # batch_size=2, in_features=3
    weight = torch.randn(4, 3)  # out_features=4, in_features=3
    output = F.linear(input_tensor, weight)
    print(f"输入形状: {input_tensor.shape}")
    print(f"权重形状: {weight.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出:\n{output}\n")

    # 手动验证: y = x @ W.T
    manual_output = input_tensor @ weight.T
    print(f"手动计算 y = x @ W.T:\n{manual_output}")
    print(f"是否相等: {torch.allclose(output, manual_output)}\n")

    # 示例 2: 带偏置
    print("示例 2: 带偏置")
    bias = torch.randn(4)  # out_features=4
    output_with_bias = F.linear(input_tensor, weight, bias)
    print(f"偏置形状: {bias.shape}")
    print(f"输出形状: {output_with_bias.shape}")
    print(f"输出:\n{output_with_bias}\n")

    # 手动验证: y = x @ W.T + b
    manual_output_with_bias = input_tensor @ weight.T + bias
    print(f"手动计算 y = x @ W.T + b:\n{manual_output_with_bias}")
    print(f"是否相等: {torch.allclose(output_with_bias, manual_output_with_bias)}\n")

    # 示例 3: 单样本输入
    print("示例 3: 单样本输入")
    single_input = torch.randn(3)  # in_features=3
    output_single = F.linear(single_input, weight, bias)
    print(f"单样本输入: {single_input}")
    print(f"单样本输入形状: {single_input.shape}")
    print(f"输出形状: {output_single.shape}")
    print(f"输出: {output_single}\n")

    # 示例 4: 高维输入 (如序列数据)
    print("示例 4: 高维输入 (序列数据)")
    seq_input = torch.randn(2, 5, 3)  # batch_size=2, seq_len=5, in_features=3
    output_seq = F.linear(seq_input, weight, bias)
    print(f"序列输入形状: {seq_input.shape}")
    print(f"输出形状: {output_seq.shape}")
    print(f"输出:\n{output_seq}\n")

if __name__ == "__main__":
    demo_f_linear()
