# 为Node1 生成启动脚本(需要指定卡数)
```python

python launch_multi_machine.py \
    --master_addr 115.190.188.193 \
    --master_port 2333 \
    --node_rank 0 \
    --world_size 10 \
    --current_node_gpus 8 \
    --model_path /data/Qwen3-8B/Qwen3-8B \
    --create_script \
    --output_script node1_launch.py

```

# 为 Node 2 生成脚本（2张卡）
python launch_multi_machine.py \
    --master_addr 115.190.188.194 \
    --master_port 2333 \
    --node_rank 1 \
    --world_size 10 \
    --current_node_gpus 2 \
    --model_path /data/Qwen3-8B/Qwen3-8B \
    --create_script \
    --output_script node2_launch.py