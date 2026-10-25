from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.history = []

    def request(self, item):
        if item in self.cache:
            self.cache.move_to_end(item)
            self.hits += 1
        else:
            if len(self.cache) >= self.cap:
                self.cache.popitem(last=False)
            self.cache[item] = 1
            self.misses += 1
        total = self.hits + self.misses
        self.history.append(self.hits / total)

class LFUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.cache = {}
        self.freq = {}
        self.hits = 0
        self.misses = 0
        self.history = []

    def request(self, item):
        if item in self.cache:
            self.freq[item] += 1
            self.hits += 1
        else:
            if len(self.cache) >= self.cap:
                evict = min(self.freq, key=self.freq.get)
                del self.cache[evict]
                del self.freq[evict]
            self.cache[item] = 1
            self.freq[item] = 1
            self.misses += 1
        total = self.hits + self.misses
        self.history.append(self.hits / total)

# workload
rng = np.random.default_rng(42)
n_items, cache_size, n_req = 100, 20, 5000
ranks = np.arange(1, n_items + 1)
weights = 1.0 / ranks; weights /= weights.sum()
requests = rng.choice(n_items, size=n_req, p=weights)

lru = LRUCache(cache_size)
lfu = LFUCache(cache_size)
for item in requests:
    lru.request(item)
    lfu.request(item)

print(f"LRU hit ratio: {lru.history[-1]:.3f}")
print(f"LFU hit ratio: {lfu.history[-1]:.3f}")

x = range(1, n_req + 1)
plt.figure(figsize=(12, 4))
plt.plot(x, [r * 100 for r in lru.history], label="LRU", color="steelblue", alpha=0.8)
plt.plot(x, [r * 100 for r in lfu.history], label="LFU", color="orange", alpha=0.8)
plt.xlabel("Requests"); plt.ylabel("Hit ratio (%)")
plt.title("Day 1: LRU vs LFU baseline")
plt.legend(); plt.tight_layout()
plt.savefig("day1_baseline.png", dpi=150); plt.show()
print("Saved day1_baseline.png")
