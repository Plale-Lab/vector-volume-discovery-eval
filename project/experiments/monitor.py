import psutil
import time
import csv
import datetime

# Configuration
OUTPUT_FILE = "logs/system_resources.csv"
INTERVAL = 1.0 

def monitor_resources():
    print(f"Starting resource monitor. Logging to {OUTPUT_FILE}...")
    
    # 1. Prime the CPU counter so the first loop value is accurate
    psutil.cpu_percent(interval=None)
    
    # 2. Initialize network counters for the differential calculation
    last_net_io = psutil.net_io_counters()
    
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "cpu_percent", "memory_percent", "net_sent_mb_s", "net_recv_mb_s"])
        
        try:
            while True:
                # Sleep first to create the interval for measurement
                time.sleep(INTERVAL)
                
                timestamp = datetime.datetime.now().isoformat()
                
                # CPU & Memory
                # interval=None is now safe because we sleep(INTERVAL) above
                cpu = psutil.cpu_percent(interval=None) 
                mem = psutil.virtual_memory().percent
                
                # Network I/O (Differential)
                net_io_now = psutil.net_io_counters()
                
                # Calculate delta (bytes since last loop)
                sent_bytes_delta = net_io_now.bytes_sent - last_net_io.bytes_sent
                recv_bytes_delta = net_io_now.bytes_recv - last_net_io.bytes_recv
                
                # Convert to MB/s (assuming INTERVAL is 1.0)
                sent_mb_s = (sent_bytes_delta / 1024 / 1024) / INTERVAL
                recv_mb_s = (recv_bytes_delta / 1024 / 1024) / INTERVAL
                
                # Reset the baseline for the next second
                last_net_io = net_io_now
                
                writer.writerow([timestamp, cpu, mem, f"{sent_mb_s:.2f}", f"{recv_mb_s:.2f}"])
                f.flush()
                
        except KeyboardInterrupt:
            print("Monitoring stopped.")

if __name__ == "__main__":
    monitor_resources()