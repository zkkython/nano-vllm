from time import sleep
import torch
import random


# 模拟生成一个生产prompt 的函数
def listen_request(config, p: float = 0.1):

    prompt = []
    prompt_len = 0
    random_p = random.randint(1, 100)
    if random_p / 100.0 <= p:
        prompt_len = random.randint(config.max_prompt_len // 4, config.max_prompt_len)
        prompt = torch.randint(0, config.vocab_size, (1, prompt_len))
        # 之所以取0，是因为(1, prompt_len) 是一个2维的，[[xx,xx,xxx]]，所以还是得取[0]
        prompt = prompt[0].tolist()

    return prompt, prompt_len


if __name__ == "__main__":

    class Config:
        max_prompt_len = 10
        vocab_size = 10000

    while True:
        config = Config()
        prompt, prompt_len = listen_request(config)
        if prompt_len > 0:
            print(prompt)
        sleep(1)
