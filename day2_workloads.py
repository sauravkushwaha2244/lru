from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

class LRUCache:
    def __init__(self, cap):
        self.cap = cap
        self.cache = OrderedDict()
        self.hits = self.misses = 0
        self.history = []
    def request(self, item):
        if item in self.cache:
            self.cache.move_to_end(item); self.hits += 1
        else:
            if len(self.cache) >= self.cap:
                self.cache.popitem(last=False)
            self.cache[item] = 1; self.misses += 1
        self.history.append(self.hits / (self.hits + self.misses))

def run(requests, cap):
    c = LRUCache(cap)
    for r in requests: c.request(r)
    return c.history

rng = np.random.default_rng(0)
n = 3000

# workload A: zipf (popular items repeat, LRU does well)
ranks = np.arange(1, 101); w = 1.0/ranks; w /= w.sum()
zipf = rng.choice(100, size=n, p=w)

# workload B: cyclic (scans 25 items in a loop, cache size 20 - LRU thrashes)
cyclic = np.tile(np.arange(25), n // 25 + 1)[:n]

# workload C: mixed (70% zipf, 30% cyclic)
mask = rng.random(n) < 0.7
mixed = np.where(mask, zipf, cyclic)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, reqs, title in zip(axes,
    [zipf, cyclic, mixed],
    ["Zipf (popular items repeat)", "Cyclic (LRU thrashes)", "Mixed"]):
    h = run(reqs, 20)
    ax.plot([r*100 for r in h], color="steelblue")
    ax.set_title(title); ax.set_xlabel("Requests")
    ax.set_ylabel("Hit ratio (%)"); ax.set_ylim(0, 100)
    ax.axhline(h[-1]*100, color="red", linestyle="--",
               label=f"Final: {h[-1]*100:.1f}%")
    ax.legend()

plt.tight_layout(); plt.savefig("day2_workloads.png", dpi=150); plt.show()
print("Saved day2_workloads.png")
print("Cyclic final hit ratio:", f"{run(cyclic, 20)[-1]*100:.1f}%",
      "- this is what the ML model must fix")
