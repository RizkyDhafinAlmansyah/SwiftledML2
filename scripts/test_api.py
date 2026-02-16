# ============================================================
# API Test Script
# ============================================================
# Tests all endpoints of the AI Engine including new buffer endpoints.
# ============================================================

import requests
import time
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("\n[1] Testing /health...")
    resp = requests.get(f"{BASE_URL}/health")
    print(f"    Status: {resp.status_code}")
    print(f"    Response: {resp.json()}")
    return resp.status_code == 200


def test_version():
    """Test version endpoint."""
    print("\n[2] Testing /version...")
    resp = requests.get(f"{BASE_URL}/version")
    print(f"    Status: {resp.status_code}")
    print(f"    Response: {resp.json()}")
    return resp.status_code == 200


def test_push_reading():
    """Test push reading endpoint."""
    print("\n[3] Testing /v1/push-reading...")
    
    readings = [
        {"temperature_c": 28.5, "humidity_rh": 78.0, "nh3_ppm": 8.0},
        {"temperature_c": 29.0, "humidity_rh": 76.0, "nh3_ppm": 9.0},
        {"temperature_c": 29.5, "humidity_rh": 74.0, "nh3_ppm": 10.0},
    ]
    
    for i, reading in enumerate(readings):
        payload = {
            "node_id": "test-node",
            **reading
        }
        resp = requests.post(f"{BASE_URL}/v1/push-reading", json=payload)
        data = resp.json()
        print(f"    Reading {i+1}: buffer_size={data.get('buffer_size', 'N/A')}")
    
    return True


def test_buffer_stats():
    """Test buffer stats endpoint."""
    print("\n[4] Testing /v1/buffer-stats/test-node...")
    resp = requests.get(f"{BASE_URL}/v1/buffer-stats/test-node")
    data = resp.json()
    print(f"    Status: {resp.status_code}")
    print(f"    Buffer size: {data.get('buffer_size', 'N/A')}")
    print(f"    Rolling stats sample: temp_avg_10min={data.get('rolling_stats', {}).get('temp_avg_10min', 'N/A')}")
    return resp.status_code == 200


def test_buffer_memory():
    """Test buffer memory endpoint."""
    print("\n[5] Testing /v1/buffer-memory...")
    resp = requests.get(f"{BASE_URL}/v1/buffer-memory")
    data = resp.json()
    print(f"    Status: {resp.status_code}")
    print(f"    Nodes: {data.get('num_nodes', 'N/A')}, Readings: {data.get('total_readings', 'N/A')}")
    print(f"    Memory: {data.get('estimated_mb', 'N/A')} MB")
    return resp.status_code == 200


def test_buffer_nodes():
    """Test buffer nodes endpoint."""
    print("\n[6] Testing /v1/buffer-nodes...")
    resp = requests.get(f"{BASE_URL}/v1/buffer-nodes")
    data = resp.json()
    print(f"    Status: {resp.status_code}")
    print(f"    Nodes: {data.get('nodes', [])}")
    return resp.status_code == 200


def test_decide_v2():
    """Test enhanced decide v2 endpoint."""
    print("\n[7] Testing /v2/decide...")
    
    payload = {
        "node_id": "test-node",
        "temperature_c": 30.0,
        "humidity_rh": 72.0,
        "nh3_ppm": 12.0,
        "use_buffer": True
    }
    
    resp = requests.post(f"{BASE_URL}/v2/decide", json=payload)
    data = resp.json()
    
    print(f"    Status: {resp.status_code}")
    print(f"    Grade: {data.get('grade', 'N/A')}")
    print(f"    Sprayer: {'ON' if data.get('sprayer_on') else 'OFF'} ({data.get('sprayer_reason', 'N/A')})")
    print(f"    Anomaly: {data.get('anomaly', {}).get('verdict', 'N/A')}")
    print(f"    Buffer size: {data.get('buffer_stats', {}).get('buffer_size', 'N/A')}")
    
    if data.get('features_used'):
        print(f"    Features used: {list(data['features_used'].keys())[:5]}...")
    
    return resp.status_code == 200


def test_legacy_decide():
    """Test legacy decide endpoint."""
    print("\n[8] Testing /decide (legacy)...")
    
    payload = {
        "node_id": "test-node-legacy",
        "temperature_c": 29.0,
        "humidity_rh": 75.0,
        "nh3_ppm": 10.0
    }
    
    resp = requests.post(f"{BASE_URL}/decide", json=payload)
    data = resp.json()
    
    print(f"    Status: {resp.status_code}")
    print(f"    Grade: {data.get('grade', 'N/A')}")
    print(f"    Sprayer: {'ON' if data.get('sprayer_on') else 'OFF'}")
    
    return resp.status_code == 200


def test_clear_buffer():
    """Test buffer clear endpoint."""
    print("\n[9] Testing /v1/buffer-clear/test-node...")
    resp = requests.post(f"{BASE_URL}/v1/buffer-clear/test-node")
    data = resp.json()
    print(f"    Status: {resp.status_code}")
    print(f"    Response: {data}")
    return resp.status_code == 200


def main():
    print("=" * 60)
    print("AI ENGINE API TEST")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    
    # Check connection first
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
    except Exception as e:
        print(f"\nERROR: Cannot connect to AI Engine at {BASE_URL}")
        print(f"Make sure it's running: python ai-engine/app.py")
        print(f"Error: {e}")
        return
    
    tests = [
        ("Health", test_health),
        ("Version", test_version),
        ("Push Reading", test_push_reading),
        ("Buffer Stats", test_buffer_stats),
        ("Buffer Memory", test_buffer_memory),
        ("Buffer Nodes", test_buffer_nodes),
        ("Decide v2", test_decide_v2),
        ("Legacy Decide", test_legacy_decide),
        ("Clear Buffer", test_clear_buffer),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, p in results if p)
    failed = len(results) - passed
    
    for name, p in results:
        status = "✓ PASS" if p else "✗ FAIL"
        print(f"  {status} - {name}")
    
    print(f"\nTotal: {passed}/{len(results)} passed")
    
    if failed == 0:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {failed} test(s) failed")


if __name__ == "__main__":
    main()
