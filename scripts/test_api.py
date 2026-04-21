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


def test_pump_ml():
    """Test ML-based pump recommendation."""
    print("\n[10] Testing /v1/recommend-pump-action (ML mode)...")

    payload = {
        "node_id": "test-pump-ml",
        "rbw_id": "rbw-01",
        "current_temp": 31.0,
        "current_humid": 70.0,
        "current_ammonia": 12.0,
        "pump_currently_on": False,
        "use_ml": True,
    }

    resp = requests.post(f"{BASE_URL}/v1/recommend-pump-action", json=payload)
    data = resp.json()

    print(f"    Status: {resp.status_code}")
    print(f"    Action: {data.get('action', 'N/A')}")
    print(f"    Reason: {data.get('reason', 'N/A')}")
    print(f"    Engine: {data.get('engine', 'N/A')}")
    print(f"    Confidence: {data.get('confidence', 'N/A')}")
    print(f"    Duration (s): {data.get('recommended_duration_seconds', 'N/A')}")

    # Engine should exist and be one of the expected values
    assert data.get("engine") in ("ml+safety", "rule-fallback", "safety", "rule-only"), \
        f"Unexpected engine: {data.get('engine')}"

    return resp.status_code == 200


def test_pump_safety_override():
    """Test that safety rules override ML when NH3 is critical."""
    print("\n[11] Testing pump safety override (NH3 critical)...")

    payload = {
        "node_id": "test-pump-safety",
        "rbw_id": "rbw-01",
        "current_temp": 25.0,
        "current_humid": 85.0,
        "current_ammonia": 30.0,   # Critical NH3 level
        "pump_currently_on": False,
        "use_ml": True,
    }

    resp = requests.post(f"{BASE_URL}/v1/recommend-pump-action", json=payload)
    data = resp.json()

    print(f"    Status: {resp.status_code}")
    print(f"    Action: {data.get('action', 'N/A')}")
    print(f"    Engine: {data.get('engine', 'N/A')}")
    print(f"    Reason: {data.get('reason', 'N/A')}")

    # Safety override: should always turn_on with engine=safety
    assert data.get("action") == "turn_on", f"Expected turn_on but got {data.get('action')}"
    assert data.get("engine") == "safety", f"Expected safety engine but got {data.get('engine')}"

    return resp.status_code == 200


def test_pump_rule_fallback():
    """Test pump with use_ml=False (pure rule-based)."""
    print("\n[12] Testing pump rule-based (use_ml=False)...")

    payload = {
        "node_id": "test-pump-rule",
        "rbw_id": "rbw-01",
        "current_temp": 31.0,
        "current_humid": 70.0,
        "current_ammonia": 8.0,
        "pump_currently_on": False,
        "use_ml": False,
    }

    resp = requests.post(f"{BASE_URL}/v1/recommend-pump-action", json=payload)
    data = resp.json()

    print(f"    Status: {resp.status_code}")
    print(f"    Action: {data.get('action', 'N/A')}")
    print(f"    Engine: {data.get('engine', 'N/A')}")

    assert data.get("engine") == "rule-only", f"Expected rule-only engine but got {data.get('engine')}"

    return resp.status_code == 200


def test_feedback():
    """Test feedback submission endpoint."""
    print("\n[13] Testing /v1/feedback...")

    payload = {
        "node_id": "test-node",
        "actual_grade": "bagus",
        "pump_was_needed": True,
        "pump_was_effective": True,
        "duration_feedback": "just_right",
        "notes": "Test feedback from test_api.py",
    }

    resp = requests.post(f"{BASE_URL}/v1/feedback", json=payload)
    data = resp.json()

    print(f"    Status: {resp.status_code}")
    if resp.status_code == 200:
        print(f"    OK: {data.get('ok', 'N/A')}")
        print(f"    Feedback ID: {data.get('feedback_id', 'N/A')}")
    elif resp.status_code == 503:
        print(f"    Expected: DB not configured ({data.get('detail', '')})")
    print(f"    Response: {data}")

    # Either 200 (DB configured) or 503 (DB not configured) is acceptable
    return resp.status_code in (200, 503)


def test_feedback_stats():
    """Test feedback stats endpoint."""
    print("\n[14] Testing /v1/feedback-stats...")

    resp = requests.get(f"{BASE_URL}/v1/feedback-stats")
    data = resp.json()

    print(f"    Status: {resp.status_code}")
    if resp.status_code == 200:
        print(f"    Total feedbacks: {data.get('total_feedbacks', 'N/A')}")
    elif resp.status_code == 503:
        print(f"    Expected: DB not configured ({data.get('detail', '')})")
    print(f"    Response: {json.dumps(data, indent=2)[:200]}...")

    return resp.status_code in (200, 503)


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
        ("Pump ML", test_pump_ml),
        ("Pump Safety Override", test_pump_safety_override),
        ("Pump Rule Fallback", test_pump_rule_fallback),
        ("Feedback Submit", test_feedback),
        ("Feedback Stats", test_feedback_stats),
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
