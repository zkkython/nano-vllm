import torch
import numpy as np

def explain_argmax():
    """
    Argmax 函数用于找到数组或张量中最大元素的索引。

    在 NumPy 中:
    - np.argmax(a, axis=None, out=None, keepdims=<no value>)
      - a: 输入数组
      - axis: 用于查找最大值的轴。如果为 None，则返回展平数组的最大元素索引
      - 返回: 整数索引或索引数组

    在 PyTorch 中:
    - torch.argmax(input, dim=None, keepdim=False, *, out=None)
      - input: 输入张量
      - dim: 用于查找最大值的维度。如果为 None，则返回展平张量的最大元素索引
      - keepdim: 如果为 True，则保持输出维度不变
      - 返回: 张量，包含最大元素的索引

    用法场景:
    - 在分类任务中，用于将 softmax 概率分布转换为预测标签
    - 在决策过程中，选择最优选项的索引
    - 机器学习中，找到最佳超参数配置的索引
    """

print("Argmax 函数解释:")
explain_argmax()

def test_argmax():
    """测试 argmax 函数的各种情况"""

    print("\n=== NumPy argmax 测试 ===")

    # 1D 数组
    arr = np.array([1, 5, 3, 7, 2])
    max_idx = np.argmax(arr)
    print(f"1D 数组 {arr}: argmax = {max_idx}, 对应的值 = {arr[max_idx]}")

    # 2D 数组
    arr_2d = np.array([[1, 2, 3], [6, 5, 4]])
    max_idx_all = np.argmax(arr_2d)  # 展平后
    max_idx_axis0 = np.argmax(arr_2d, axis=0)  # 每列最大元素索引
    max_idx_axis1 = np.argmax(arr_2d, axis=1)  # 每行最大元素索引
    print(f"2D 数组:\n{arr_2d}")
    print(f"argmax (展平): {max_idx_all}")
    print(f"argmax (axis=0): {max_idx_axis0}")
    print(f"argmax (axis=1): {max_idx_axis1}")

    print("\n=== PyTorch argmax 测试 ===")

    # 1D 张量
    tensor = torch.tensor([1, 5, 3, 7, 2])
    max_idx = torch.argmax(tensor)
    print(f"1D 张量 {tensor}: argmax = {max_idx}, 对应的值 = {tensor[max_idx]}")

    # 2D 张量
    tensor_2d = torch.tensor([[1, 2, 3], [6, 5, 4]], dtype=torch.float32)
    max_idx_all = torch.argmax(tensor_2d)  # 展平后
    max_idx_dim0 = torch.argmax(tensor_2d, dim=0)  # 每列最大元素索引
    max_idx_dim1 = torch.argmax(tensor_2d, dim=1)  # 每行最大元素索引
    print(f"2D 张量:\n{tensor_2d}")
    print(f"argmax (展平): {max_idx_all}")
    print(f"argmax (dim=0): {max_idx_dim0}")
    print(f"argmax (dim=1): {max_idx_dim1}")

    # 模拟分类任务
    print("\n=== 分类任务示例 ===")
    logits = torch.randn(3, 4)  # 假设 3 个样本，4 个类别
    predicted_labels = torch.argmax(logits, dim=1)
    print(f"模型预测 logits:\n{logits}")
    print(f"预测的 labels (argmax 结果): {predicted_labels}")

if __name__ == "__main__":
    test_argmax()
