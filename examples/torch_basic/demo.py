from re import I
import torch


torch.manual_seed(42)
a = torch.full(size=(1,), fill_value=-1)
print(a.item())

b = torch.ones(1)
print(b)


d = []
d.extend([1, 23])
d.extend([4, 5])
print(d)


print([-1] * 5)

print("/root/model/deepseek".split("/")[-1])

print("*" * 70)
expert_hit = torch.randint(5, (5, 1, 2))
print(expert_hit)
g = torch.greater(expert_hit.sum(dim=(-1, -2)), 2)
print(f"g sgape {g.shape}")
gnot0 = g.nonzero()
print(f"gnot0 sgape {gnot0.shape}")
for ex_id in gnot0:
    print(f"before {ex_id}")
    ex_id = ex_id.item()
    print(ex_id)
    print(expert_hit[ex_id])

print("*" * 100)

import torch.nn.functional as F

hidden_states = torch.randint(5, (10, 1024), dtype=torch.float)

router_logits = F.linear(hidden_states, weight=torch.rand(5, 1024), bias=None)
num_experts_per_tok = 3

print(f"origin {router_logits}")


def route_tokens_to_experts(router_logits):
    # (batch_size_sequence_length, num_experts) 做相关性计算
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
    # (batch_size_sequence_length, num_experts) 选择topk， 从所有专家里给每一个token选择topk个专家
    # routing_weights： (batch_size_sequence_length, self.num_experts_per_tok)
    # selected_experts: (batch_size_sequence_length, self.num_experts_per_tok)
    routing_weights, selected_experts = torch.topk(
        routing_weights, num_experts_per_tok, dim=-1
    )
    # 是否做归一化
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(router_logits.dtype)
    return selected_experts, routing_weights


selected_experts, routing_weights = route_tokens_to_experts(router_logits)

# hidden = torch.randint(5, (10, 5))
# print(f"origin {hidden}")
topk_weight, top_k_index = routing_weights, selected_experts
# topk_weight, topk_index = torch.topk(hidden, k=3, dim=-1)
# 10个token, top3

print(f"top k index {top_k_index} \n top k weight: {topk_weight}")
# 5 个专家，(10, 3) -> (10, 3, 5) -> (5, 3, 10), 这是个0，1的矩阵
expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=5).permute(2, 1, 0)
print(f"expert mask {expert_mask}")

# 专家被tokens选择的次数, （5，）
experts_selected_times = torch.sum(expert_mask, dim=(-2, -1))
print(f" 每个专家被选择的次数： experts selected times {experts_selected_times}")

# (5,), 返回的是第一个bool矩阵
experts_selected_times_above_5_bool = torch.greater(experts_selected_times, 5)
print(f"专家被选择次数>0的专家有哪些{experts_selected_times_above_5_bool}")

# (True的个数, 1)
expert_hit = experts_selected_times_above_5_bool.nonzero()
print(f"expert hit被命中{expert_hit}")

# (batch_size_sequence_length, hidden_size)
final_hidden_states = torch.zeros_like(hidden_states)
for expert_idx in expert_hit:
    expert_idx = expert_idx.item()  # 取出标量
    # 获取某个专家的(topk, num_tokens)
    expert_idx_topk_tokens = expert_mask[expert_idx]
    print(f"expert idx {expert_idx}, topk and tokens {expert_idx_topk_tokens}")
    idx, top_x = torch.where(expert_idx_topk_tokens)
    print(f"idx {idx}, top_x {top_x}")
    # (selected_tokens, hidden_dim)
    num_selected_tokens = hidden_states[top_x]
    print(f"selected tokens {num_selected_tokens}")

    current_state = num_selected_tokens

    current_hidden_states = (
        F.linear(current_state, weight=torch.rand(1024, 1024))
    ) * topk_weight[top_x, idx, None]
    print(f"current_hidden_states {current_hidden_states}")

    final_hidden_states.index_add_(
        dim=0,
        index=top_x,
        source=current_hidden_states,
    )


print(f"final hidden states {final_hidden_states}")
