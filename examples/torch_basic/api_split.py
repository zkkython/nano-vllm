import torch


# split on dim=0 (rows)
def test_split_int():
    x = torch.arange(12).reshape(6, 2)  # shape: (6, 2), 6行

    # 每块3行分割
    chunks = x.split(3, dim=0)

    assert len(chunks) == 2
    assert chunks[0].shape == (3, 2)  # [[0,1],[2,3],[4,5]]
    assert chunks[1].shape == (3, 2)  # [[6,7],[8,9],[10,11]]

    print("✓ split(int) test passed")
    return chunks


# 使用list[int]自定义分割
def test_split_list():
    x = torch.arange(10).reshape(5, 2)  # shape: (5, 2), 5行

    # [1,2,2] 表示第一块1行，后两块各2行
    chunks = x.split([1, 2, 2], dim=0)

    assert len(chunks) == 3
    assert chunks[0].shape == (1, 2)  # [[0,1]]
    assert chunks[1].shape == (2, 2)  # [[2,3],[4,5]]
    assert chunks[2].shape == (2, 2)  # [[6,7],[8,9]]

    print("✓ split(list) test passed")
    return chunks


# split on different dim， dim = 1 (columns)
def test_split_diff_dim():
    x = torch.arange(24).reshape(3, 8)  # shape: (3, 8), 8列

    # 每块4列分割
    chunks = x.split(4, dim=1)

    assert len(chunks) == 2
    assert all(c.shape[0] == 3 for c in chunks)  # 行数不变
    assert chunks[0].shape == (3, 4)  # 前4列
    assert chunks[1].shape == (3, 4)  # 后4列

    print("✓ split different dim test passed")
    return chunks


# 不均匀分割
def test_split_uneven():
    x = torch.arange(9).reshape(9, 1)  # shape: (9, 1), 9行

    chunks = x.split([2, 3, 4], dim=0)  # 2+3+4=9

    assert len(chunks) == 3
    assert chunks[0].shape == (2, 1)
    assert chunks[1].shape == (3, 1)
    assert chunks[2].shape == (4, 1)

    print("✓ split uneven test passed")
    return chunks


if __name__ == "__main__":
    import torch

    test_split_int()
    test_split_list()
    test_split_diff_dim()
    test_split_uneven()

    print("All split tests passed!")
