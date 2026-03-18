from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256 # 每个内存块（block）能容纳 256 个 token
    counter = count() # 全局序列计数器，用于分配唯一的 seq_id (是类变量，不是实例变量，所有实例共享)

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids) # 当前总 token 数 ( = prompt + completion )
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = [] # 指向物理内存块的索引列表
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self): # 支持len(seq)操作
        return self.num_tokens

    def __getitem__(self, key): # 支持seq[i]操作
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self): # 已生成的 token 数量
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self): # 已缓存的完整 block 数量
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self): # 当前序列总共需要多少个 block（向上取整）
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self): # 最后一个 block 中实际使用的 token 数（可能不满 256）
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i): # 返回第 i 个 block 对应的 token 片段（用于调试或验证）
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int): # 向序列追加一个新生成的 token
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # 支持 pickle 序列化，以便在多进程间传递 Sequence 对象
    # 添加优化：当有 completion tokens 时，不保存整个 token_ids 列表，只保存最后一个 token，以节省内存和带宽
    # （因为前面的 tokens 可能已通过 KV Cache 缓存，无需重复传输）
    def __getstate__(self): # 序列化
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)
    def __setstate__(self, state): # 反序列化
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
