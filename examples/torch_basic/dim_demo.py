import torch

def explain_dim():
    """
    在 PyTorch 中，dim 参数指定了操作沿着哪个维度进行。
    想象一个多维数组就像一叠纸：

    - 0维 (标量): 就像一小片纸。
    - 1维 (向量): 一行纸片。
    - 2维 (矩阵): 一叠纸片，形如 rectangle(矩形)。
    - 3维 (张量): 一叠 rectangle，就像一堆货架上的书。

    dim参数工作方式：
    - dim=0: 沿着纸片的堆叠方向（像垂直穿过纸片的堆迭）
    - dim=1: 沿着纸片的左右方向（水平穿透纸片的宽度）
    - dim=2: 沿着纸片的前后方向（水平穿透纸片的深度）

    对于3D张量 [batch, row, col]:
    - dim=0: 沿着 batch 维度，处理每个样本控制来自不同像素。
    - dim=1: 沿着 row 维度，处理每个行中的像素。
    - dim=2: 沿着 col 维度，在同一行内从左到右移动。
    """

    print("PyTorch 中 dim 参数的理解方法：")
    print(explain_dim.__doc__)

def demo_dim_operations():
    """演示不同维度上的操作"""

    print("\n=== 创建3D张量示例 ===")
    # shape: (2, 3, 4) - 想象成2个3行4列的矩阵堆叠
    tensor = torch.tensor([
        [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12]],  # 第1个矩阵

        [[13, 14, 15, 16],
         [17, 18, 19, 20],
         [21, 22, 23, 24]]   # 第2个矩阵
    ], dtype=torch.float32)
    print(f"原始3D张量 shape: {tensor.shape}")
    print(f"张量内容:\n{tensor}")

    print("\n=== 操作演示 ===")

    # sum 操作
    print("\n1. torch.sum - 求和")
    print(f"torch.sum(tensor) - 所有元素求和: {torch.sum(tensor)}")
    print(f"torch.sum(tensor, dim=0) - 沿着第一个维度求和 shape: {torch.sum(tensor, dim=0).shape}")
    print(f"结果:\n{torch.sum(tensor, dim=0)}")
    print(f"torch.sum(tensor, dim=1) - 沿着第二个维度求和 shape: {torch.sum(tensor, dim=1).shape}")
    print(f"结果:\n{torch.sum(tensor, dim=1)}")
    print(f"torch.sum(tensor, dim=2) - 沿着第三个维度求和 shape: {torch.sum(tensor, dim=2).shape}")
    print(f"结果:\n{torch.sum(tensor, dim=2)}")

    # mean 操作
    print("\n2. torch.mean - 平均值")
    print(f"torch.mean(tensor, dim=0) - 第一个维度的平均 shape: {torch.mean(tensor, dim=0).shape}")
    print(f"结果:\n{torch.mean(tensor, dim=0)}")

    # argmax 操作
    print("\n3. torch.argmax - 最大值索引")
    print(f"torch.argmax(tensor, dim=0) - 第一个维度的最大值索引:")
    print(f"结果索引:\n{torch.argmax(tensor, dim=0)}")
    print("解释: 对于每个(row, col)位置，返回两个矩阵中哪个矩阵的值更大")

    print(f"\ntorch.argmax(tensor, dim=1) - 第二个维度的最大值索引:")
    print(f"结果索引:\n{torch.argmax(tensor, dim=1)}")
    print("解释: 对于每个(batch, col)位置，返回三行中哪一行的值最大")

    print(f"\ntorch.argmax(tensor, dim=2) - 第三个维度的最大值索引:")
    print(f"结果索引:\n{torch.argmax(tensor, dim=2)}")
    print("解释: 对于每个(batch, row)位置，返回四列中哪一列的值最大")

    # max 操作 (返回值和索引)
    print("\n4. torch.max - 最大值和索引")
    max_val, max_idx = torch.max(tensor, dim=1)
    print(f"torch.max(tensor, dim=1) - 第二个维度的最大值 shape: {max_val.shape}")
    print(f"最大值:\n{max_val}")
    print(f"对应索引:\n{max_idx}")

    print("\n理解方法总结:")
    print("- dim=0: 对最外层的包进行操作")
    print("- dim=1: 对中间层进行操作 (通常是特征或序列长度)")
    print("- dim=2: 对最内层进行操作 (比如图像的像素列)")
    print("\n你也可以想成:")
    print("- dim=0: '叠加在一起'")
    print("- dim=1: '水平展开'")
    print("- dim=2: '垂直跨越'")

if __name__ == "__main__":
    demo_dim_operations()
