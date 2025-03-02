import random
import torch

class CustomDeque:
    def __init__(self, max_len):
        self.max_len = max_len
        self.buffer = []
    
    def push(self, item):
        if len(self.buffer) >= self.max_len:
            self.buffer.pop(0)  
        self.buffer.append(item)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = CustomDeque(capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.push((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return self.buffer.sample(batch_size)

    def size(self):
        return self.buffer.size()
    
    def clear(self):
        self.buffer.clear()

