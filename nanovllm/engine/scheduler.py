from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]: # 返回 scheduled_seqs 和 is_prefill
        scheduled_seqs = []

        # prefill
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0] # 从 waiting 队列头部尝试调度新序列
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq): # 检查两个限制
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True # Prefill 优先策略：只要有新序列能调度，就优先处理 prefill

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft() # 从 running 队列逐个取出序列尝试调度
            while not self.block_manager.can_append(seq): # 对每个序列，检查其是否还能追加新 token（内存块足够）
                # 如果不能（内存不足），则抢占其他序列来腾出空间
                if self.running: # 优先抢占 running 队列尾部的序列
                    self.preempt(self.running.pop())
                else: # running 队列已空，则抢占自己
                    self.preempt(seq)
                    break
            else: # 能追加新 token，则调度该序列
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence): # 抢占指定序列的资源，放回 waiting 队列
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq) # 释放其占用的内存块
        self.waiting.appendleft(seq) # 放入 waiting 队列头部，以便尽快重新调度（避免饥饿）

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id) # 将模型生成的 token_ids 追加到对应序列
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens: # 检查是否应终止序列
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
