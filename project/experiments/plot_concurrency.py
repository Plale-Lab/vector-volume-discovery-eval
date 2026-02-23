import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Configuration ---
RESULTS_FILE = "logs/qdrant_concurrency_results.csv"
CHART_FILE = "logs/concurrency_chart.png"

# --- Step 1: Create the Data (Based on your latest results) ---
os.makedirs("logs", exist_ok=True)

data = {
    "num_workers": [1, 10, 100, 1000, 10000, 100000, 1000000],
    "throughput_qps": [22124.94, 6136.56, 3620.90, 3636.97, 3449.42, 3556.53, 3579.42],
    "p99_latency_ms": [0.337, 3.232, 21.893, 18.877, 20.720, 20.194, 21.872]
}

df = pd.DataFrame(data)
df.to_csv(RESULTS_FILE, index=False)
print(f"Data saved to {RESULTS_FILE}")

# --- Step 2: Plotting ---
# Create a 2-axis plot
fig, ax1 = plt.subplots(figsize=(12, 7))

# Set X-axis to Logarithmic Scale (Crucial for 1 to 1M range)
ax1.set_xscale('log')
ax1.set_xlabel('Number of Concurrent Users (Workers) - Log Scale')

# Axis 1: Latency (P99)
color = 'tab:red'
ax1.set_ylabel('P99 Latency (ms)', color=color)
line1 = ax1.plot(df["num_workers"], df["p99_latency_ms"], color=color, marker='o', linewidth=2, label="P99 Latency")
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(bottom=0, top=max(df["p99_latency_ms"]) * 1.2)  # Add some headroom

# Axis 2: Throughput (QPS)
ax2 = ax1.twinx() # Share the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Throughput (Queries per Second)', color=color)
line2 = ax2.plot(df["num_workers"], df["throughput_qps"], color=color, marker='s', linestyle='--', linewidth=2, label="Throughput (QPS)")
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(bottom=0, top=max(df["throughput_qps"]) * 1.2)

# Combine legends from both axes
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2)

# Final Touches
fig.suptitle("Qdrant Concurrency Stress Test: Latency vs. Throughput", y=0.95, fontsize=16)
plt.grid(True, which="both", linestyle='--', alpha=0.7)

# Explicitly set x-ticks to match your data points for clarity
plt.xticks(df["num_workers"], [str(x) for x in df["num_workers"]])

plt.tight_layout()
plt.savefig(CHART_FILE)

print(f"Concurrency chart saved to {CHART_FILE}")