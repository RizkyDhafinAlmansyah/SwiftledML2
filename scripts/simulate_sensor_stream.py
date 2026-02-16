# ============================================================
# Sensor Stream Simulator
# ============================================================
# Simulates sensor readings being pushed to AI Engine every minute.
# Useful for testing buffer functionality and rolling features.
# ============================================================

import argparse
import time
import random
import requests
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Simulate sensor stream")
    parser.add_argument("--node-id", default="test-node-01", help="Node ID")
    parser.add_argument("--interval", type=int, default=60, help="Interval in seconds")
    parser.add_argument("--count", type=int, default=60, help="Number of readings to send (0=infinite)")
    parser.add_argument("--url", default="http://localhost:8000", help="AI Engine URL")
    parser.add_argument("--fast", action="store_true", help="Fast mode (1 sec interval)")
    return parser.parse_args()


def generate_reading(base_temp=28, base_humid=78, base_nh3=8, variation=True):
    """Generate realistic sensor readings with optional variation."""
    if variation:
        temp = base_temp + random.uniform(-2, 3) + random.gauss(0, 0.5)
        humid = base_humid + random.uniform(-5, 5) + random.gauss(0, 1)
        nh3 = base_nh3 + random.uniform(-1, 2) + random.gauss(0, 0.3)
    else:
        temp = base_temp
        humid = base_humid
        nh3 = base_nh3
    
    # Clamp to realistic ranges
    temp = max(20, min(40, temp))
    humid = max(50, min(100, humid))
    nh3 = max(0, min(50, nh3))
    
    return round(temp, 2), round(humid, 2), round(nh3, 2)


def push_reading(url, node_id, temp, humid, nh3):
    """Push reading to AI Engine."""
    endpoint = f"{url}/v1/push-reading"
    payload = {
        "node_id": node_id,
        "temperature_c": temp,
        "humidity_rh": humid,
        "nh3_ppm": nh3,
        "timestamp": time.time()
    }
    
    try:
        resp = requests.post(endpoint, json=payload, timeout=10)
        return resp.status_code, resp.json()
    except Exception as e:
        return 0, {"error": str(e)}


def decide_v2(url, node_id, temp, humid, nh3):
    """Call v2/decide endpoint."""
    endpoint = f"{url}/v2/decide"
    payload = {
        "node_id": node_id,
        "temperature_c": temp,
        "humidity_rh": humid,
        "nh3_ppm": nh3,
        "use_buffer": True
    }
    
    try:
        resp = requests.post(endpoint, json=payload, timeout=10)
        return resp.status_code, resp.json()
    except Exception as e:
        return 0, {"error": str(e)}


def main():
    args = parse_args()
    
    interval = 1 if args.fast else args.interval
    count = args.count
    
    print("=" * 60)
    print("SENSOR STREAM SIMULATOR")
    print("=" * 60)
    print(f"Node ID: {args.node_id}")
    print(f"AI Engine URL: {args.url}")
    print(f"Interval: {interval}s")
    print(f"Count: {'infinite' if count == 0 else count}")
    print("=" * 60)
    
    # Test connection
    print("\nTesting connection...")
    try:
        resp = requests.get(f"{args.url}/health", timeout=5)
        if resp.status_code == 200:
            print(f"  ✓ Connected to AI Engine: {resp.json()}")
        else:
            print(f"  ✗ Connection failed: {resp.status_code}")
            return
    except Exception as e:
        print(f"  ✗ Cannot connect: {e}")
        print("  Make sure AI Engine is running: python ai-engine/app.py")
        return
    
    print("\nStarting simulation...")
    print("-" * 60)
    
    i = 0
    try:
        while count == 0 or i < count:
            i += 1
            temp, humid, nh3 = generate_reading()
            
            # Push to buffer and get decision
            status, response = decide_v2(args.url, args.node_id, temp, humid, nh3)
            
            ts = datetime.now().strftime("%H:%M:%S")
            
            if status == 200:
                grade = response.get("grade", "?")
                sprayer = "ON" if response.get("sprayer_on") else "OFF"
                buffer_size = response.get("buffer_stats", {}).get("buffer_size", 0)
                
                print(f"[{ts}] #{i:03d} | T:{temp:5.1f}°C | H:{humid:5.1f}% | NH3:{nh3:5.1f}ppm | "
                      f"Grade:{grade:6s} | Sprayer:{sprayer} | Buffer:{buffer_size}")
            else:
                print(f"[{ts}] #{i:03d} | Error: {response}")
            
            if count == 0 or i < count:
                time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user.")
    
    print("\n" + "=" * 60)
    print(f"Simulation complete. Sent {i} readings.")
    print("=" * 60)


if __name__ == "__main__":
    main()
