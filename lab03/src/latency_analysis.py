
import torch
import time
import statistics

from models import MLP, CNN, AttentionModel

models = {
    "MLP": MLP(),
    "CNN": CNN(),
    "Attention": AttentionModel()
}

def measure_latency(model, batch_size, runs):
    model.eval()

    # Prepare input before starting the timer
    sample = torch.randn(batch_size, 1, 28, 28)

    # Warm-up runs
    with torch.inference_mode():
        for _ in range(10):
            model(sample)

        # Actual measurements
        times = []

        for _ in range(runs):
            start = time.perf_counter()
            model(sample)
            end = time.perf_counter()

            times.append((end - start) * 1000)

    return statistics.median(times)


print("Inference Latency Analysis")
print("=" * 55)

for name, model in models.items():
    latency_batch1 = measure_latency(model, batch_size=1, runs=100)
    latency_batch128 = measure_latency(model, batch_size=128, runs=20)

    print(f"\n{name}:")
    print(f"Batch size 1: {latency_batch1:.4f} ms")
    print(f"Batch size 128: {latency_batch128:.4f} ms")
