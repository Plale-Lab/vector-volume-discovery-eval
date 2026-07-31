import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

def plot_system_resources():
    input_file = "logs/system_resources.csv"
    output_image = "logs/system_resources_plot.png"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run monitor.py first.")
        return

    try:
        # 1. Load Data
        df = pd.read_csv(input_file)
        
        # Convert timestamp string to actual datetime objects for plotting
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 2. Setup Plot (2 Rows, 1 Column)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # --- Subplot 1: CPU & Memory Usage ---
        ax1.set_title("System Resource Usage During Load Test")
        ax1.plot(df['timestamp'], df['cpu_percent'], label='CPU (%)', color='#1f77b4', linewidth=2)
        ax1.plot(df['timestamp'], df['memory_percent'], label='Memory (%)', color='#ff7f0e', linewidth=2)
        ax1.set_ylabel("Usage (%)")
        ax1.set_ylim(0, 105)  # Fix y-axis to 0-100%
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(loc='upper left', frameon=True)
        
        # --- Subplot 2: Network I/O ---
        ax2.set_title("Network I/O Speed")
        ax2.plot(df['timestamp'], df['net_sent_mb_s'], label='Upload (MB/s)', color='#2ca02c', linewidth=1.5)
        ax2.plot(df['timestamp'], df['net_recv_mb_s'], label='Download (MB/s)', color='#d62728', linewidth=1.5)
        ax2.set_ylabel("Throughput (MB/s)")
        ax2.set_xlabel("Time")
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(loc='upper left', frameon=True)
        
        # 3. Format Time Axis
        # Shows Hour:Minute:Second
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)
        
        # 4. Save and Show
        plt.tight_layout()
        plt.savefig(output_image)
        print(f"Graph successfully saved to: {output_image}")
        # plt.show() # Uncomment if running locally with a display

    except Exception as e:
        print(f"An error occurred while plotting: {e}")

if __name__ == "__main__":
    plot_system_resources()