# ============================================================
# Swiftlet AI Engine - Sliding Window Buffer Manager
# ============================================================
# Manages in-memory sliding window buffers for each sensor node.
# Computes rolling statistics (avg, delta, std, trend) in real-time.
# ============================================================

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
import numpy as np


@dataclass
class SensorReading:
    """Single sensor reading with timestamp."""
    timestamp: float  # Unix epoch
    temperature: float
    humidity: float
    ammonia: float
    node_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "ammonia": self.ammonia,
            "node_id": self.node_id,
        }


@dataclass
class RollingStats:
    """Computed rolling statistics for a sensor."""
    # Averages at different windows
    temp_avg_10min: float = 0.0
    temp_avg_30min: float = 0.0
    temp_avg_60min: float = 0.0
    
    humid_avg_10min: float = 0.0
    humid_avg_30min: float = 0.0
    humid_avg_60min: float = 0.0
    
    nh3_avg_10min: float = 0.0
    nh3_avg_30min: float = 0.0
    nh3_avg_60min: float = 0.0
    
    # Deltas (percentage change)
    temp_delta_pct_10min: float = 0.0
    humid_delta_pct_10min: float = 0.0
    nh3_delta_pct_10min: float = 0.0
    
    # Standard deviations (volatility)
    temp_std_30min: float = 0.0
    humid_std_30min: float = 0.0
    nh3_std_30min: float = 0.0
    
    # Trends: "rising", "falling", "stable"
    temp_trend: str = "stable"
    humid_trend: str = "stable"
    nh3_trend: str = "stable"
    
    # Latest values
    latest_temp: float = 0.0
    latest_humid: float = 0.0
    latest_ammonia: float = 0.0
    
    # Buffer info
    buffer_size: int = 0
    oldest_reading_age_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "temp_avg_10min": round(self.temp_avg_10min, 2),
            "temp_avg_30min": round(self.temp_avg_30min, 2),
            "temp_avg_60min": round(self.temp_avg_60min, 2),
            "humid_avg_10min": round(self.humid_avg_10min, 2),
            "humid_avg_30min": round(self.humid_avg_30min, 2),
            "humid_avg_60min": round(self.humid_avg_60min, 2),
            "nh3_avg_10min": round(self.nh3_avg_10min, 2),
            "nh3_avg_30min": round(self.nh3_avg_30min, 2),
            "nh3_avg_60min": round(self.nh3_avg_60min, 2),
            "temp_delta_pct_10min": round(self.temp_delta_pct_10min, 2),
            "humid_delta_pct_10min": round(self.humid_delta_pct_10min, 2),
            "nh3_delta_pct_10min": round(self.nh3_delta_pct_10min, 2),
            "temp_std_30min": round(self.temp_std_30min, 3),
            "humid_std_30min": round(self.humid_std_30min, 3),
            "nh3_std_30min": round(self.nh3_std_30min, 3),
            "temp_trend": self.temp_trend,
            "humid_trend": self.humid_trend,
            "nh3_trend": self.nh3_trend,
            "latest_temp": round(self.latest_temp, 2),
            "latest_humid": round(self.latest_humid, 2),
            "latest_ammonia": round(self.latest_ammonia, 2),
            "buffer_size": self.buffer_size,
            "oldest_reading_age_sec": round(self.oldest_reading_age_sec, 1),
        }


class SlidingWindowBuffer:
    """
    Sliding window buffer for a single sensor node.
    Stores up to `max_readings` (default 60 = 1 hour at 1 reading/min).
    Thread-safe with a lock.
    """
    
    def __init__(self, node_id: str, max_readings: int = 60):
        self.node_id = node_id
        self.max_readings = max_readings
        self.readings: deque = deque(maxlen=max_readings)
        self.lock = threading.Lock()
        self.last_update: float = 0.0
    
    def append(self, reading: SensorReading) -> None:
        """Add a new reading to the buffer."""
        with self.lock:
            self.readings.append(reading)
            self.last_update = time.time()
    
    def get_readings(self, last_n: Optional[int] = None) -> List[SensorReading]:
        """Get readings from the buffer. If last_n specified, get only last N readings."""
        with self.lock:
            if last_n is None:
                return list(self.readings)
            return list(self.readings)[-last_n:]
    
    def get_readings_in_window(self, window_seconds: float) -> List[SensorReading]:
        """Get readings within the last `window_seconds`."""
        with self.lock:
            now = time.time()
            cutoff = now - window_seconds
            return [r for r in self.readings if r.timestamp >= cutoff]
    
    def size(self) -> int:
        """Get current buffer size."""
        with self.lock:
            return len(self.readings)
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self.lock:
            self.readings.clear()
            self.last_update = 0.0
    
    def compute_stats(self) -> RollingStats:
        """
        Compute rolling statistics from the buffer.
        Returns RollingStats with all computed features.
        """
        with self.lock:
            readings = list(self.readings)
        
        if not readings:
            return RollingStats()
        
        now = time.time()
        
        # Separate readings into time windows
        readings_10min = [r for r in readings if (now - r.timestamp) <= 600]   # 10 min
        readings_30min = [r for r in readings if (now - r.timestamp) <= 1800]  # 30 min
        readings_60min = readings  # All readings (up to 60 min)
        
        stats = RollingStats()
        
        # Latest values
        latest = readings[-1]
        stats.latest_temp = latest.temperature
        stats.latest_humid = latest.humidity
        stats.latest_ammonia = latest.ammonia
        stats.buffer_size = len(readings)
        
        if readings:
            oldest = readings[0]
            stats.oldest_reading_age_sec = now - oldest.timestamp
        
        # Compute averages
        if readings_10min:
            temps = [r.temperature for r in readings_10min]
            humids = [r.humidity for r in readings_10min]
            nh3s = [r.ammonia for r in readings_10min]
            stats.temp_avg_10min = np.mean(temps)
            stats.humid_avg_10min = np.mean(humids)
            stats.nh3_avg_10min = np.mean(nh3s)
        else:
            stats.temp_avg_10min = latest.temperature
            stats.humid_avg_10min = latest.humidity
            stats.nh3_avg_10min = latest.ammonia
        
        if readings_30min:
            temps = [r.temperature for r in readings_30min]
            humids = [r.humidity for r in readings_30min]
            nh3s = [r.ammonia for r in readings_30min]
            stats.temp_avg_30min = np.mean(temps)
            stats.humid_avg_30min = np.mean(humids)
            stats.nh3_avg_30min = np.mean(nh3s)
            
            # Standard deviations
            if len(temps) > 1:
                stats.temp_std_30min = np.std(temps)
                stats.humid_std_30min = np.std(humids)
                stats.nh3_std_30min = np.std(nh3s)
        else:
            stats.temp_avg_30min = stats.temp_avg_10min
            stats.humid_avg_30min = stats.humid_avg_10min
            stats.nh3_avg_30min = stats.nh3_avg_10min
        
        if readings_60min:
            temps = [r.temperature for r in readings_60min]
            humids = [r.humidity for r in readings_60min]
            nh3s = [r.ammonia for r in readings_60min]
            stats.temp_avg_60min = np.mean(temps)
            stats.humid_avg_60min = np.mean(humids)
            stats.nh3_avg_60min = np.mean(nh3s)
        else:
            stats.temp_avg_60min = stats.temp_avg_30min
            stats.humid_avg_60min = stats.humid_avg_30min
            stats.nh3_avg_60min = stats.nh3_avg_30min
        
        # Compute deltas (percentage change over 10 min)
        if len(readings_10min) >= 2:
            first = readings_10min[0]
            last = readings_10min[-1]
            
            if first.temperature != 0:
                stats.temp_delta_pct_10min = ((last.temperature - first.temperature) / first.temperature) * 100
            if first.humidity != 0:
                stats.humid_delta_pct_10min = ((last.humidity - first.humidity) / first.humidity) * 100
            if first.ammonia != 0:
                stats.nh3_delta_pct_10min = ((last.ammonia - first.ammonia) / first.ammonia) * 100
            
            # Clip extreme values
            stats.temp_delta_pct_10min = np.clip(stats.temp_delta_pct_10min, -100, 100)
            stats.humid_delta_pct_10min = np.clip(stats.humid_delta_pct_10min, -100, 100)
            stats.nh3_delta_pct_10min = np.clip(stats.nh3_delta_pct_10min, -100, 100)
        
        # Compute trends
        stats.temp_trend = self._compute_trend([r.temperature for r in readings_10min])
        stats.humid_trend = self._compute_trend([r.humidity for r in readings_10min])
        stats.nh3_trend = self._compute_trend([r.ammonia for r in readings_10min])
        
        return stats
    
    def _compute_trend(self, values: List[float], threshold_pct: float = 2.0) -> str:
        """
        Compute trend from a list of values.
        Returns "rising", "falling", or "stable".
        """
        if len(values) < 2:
            return "stable"
        
        first = values[0]
        last = values[-1]
        
        if first == 0:
            return "stable"
        
        pct_change = ((last - first) / first) * 100
        
        if pct_change > threshold_pct:
            return "rising"
        elif pct_change < -threshold_pct:
            return "falling"
        else:
            return "stable"


class BufferManager:
    """
    Manages sliding window buffers for multiple sensor nodes.
    Singleton-like pattern for global access.
    """
    
    def __init__(self, max_readings_per_node: int = 60):
        self.max_readings = max_readings_per_node
        self.buffers: Dict[str, SlidingWindowBuffer] = {}
        self.lock = threading.Lock()
    
    def get_buffer(self, node_id: str) -> SlidingWindowBuffer:
        """Get or create a buffer for a node."""
        with self.lock:
            if node_id not in self.buffers:
                self.buffers[node_id] = SlidingWindowBuffer(
                    node_id=node_id,
                    max_readings=self.max_readings
                )
            return self.buffers[node_id]
    
    def push_reading(
        self,
        node_id: str,
        temperature: float,
        humidity: float,
        ammonia: float,
        timestamp: Optional[float] = None
    ) -> RollingStats:
        """
        Push a new reading to the buffer and return computed stats.
        This is the main entry point for real-time data.
        """
        if timestamp is None:
            timestamp = time.time()
        
        reading = SensorReading(
            timestamp=timestamp,
            temperature=temperature,
            humidity=humidity,
            ammonia=ammonia,
            node_id=node_id
        )
        
        buffer = self.get_buffer(node_id)
        buffer.append(reading)
        
        return buffer.compute_stats()
    
    def get_stats(self, node_id: str) -> RollingStats:
        """Get current rolling stats for a node without pushing new data."""
        buffer = self.get_buffer(node_id)
        return buffer.compute_stats()
    
    def get_all_nodes(self) -> List[str]:
        """Get list of all tracked node IDs."""
        with self.lock:
            return list(self.buffers.keys())
    
    def get_node_info(self, node_id: str) -> Dict[str, Any]:
        """Get info about a specific node's buffer."""
        buffer = self.get_buffer(node_id)
        stats = buffer.compute_stats()
        return {
            "node_id": node_id,
            "buffer_size": buffer.size(),
            "max_readings": buffer.max_readings,
            "last_update": buffer.last_update,
            "stats": stats.to_dict(),
        }
    
    def clear_node(self, node_id: str) -> bool:
        """Clear buffer for a specific node."""
        with self.lock:
            if node_id in self.buffers:
                self.buffers[node_id].clear()
                return True
            return False
    
    def clear_all(self) -> int:
        """Clear all buffers. Returns number of buffers cleared."""
        with self.lock:
            count = len(self.buffers)
            for buffer in self.buffers.values():
                buffer.clear()
            return count
    
    def memory_usage_estimate(self) -> Dict[str, Any]:
        """Estimate memory usage of all buffers."""
        with self.lock:
            total_readings = sum(b.size() for b in self.buffers.values())
            # Rough estimate: ~100 bytes per reading
            estimated_bytes = total_readings * 100
            return {
                "num_nodes": len(self.buffers),
                "total_readings": total_readings,
                "estimated_bytes": estimated_bytes,
                "estimated_mb": round(estimated_bytes / (1024 * 1024), 2),
            }


# Global buffer manager instance
_buffer_manager: Optional[BufferManager] = None


def get_buffer_manager(max_readings: int = 60) -> BufferManager:
    """Get the global buffer manager instance (singleton pattern)."""
    global _buffer_manager
    if _buffer_manager is None:
        _buffer_manager = BufferManager(max_readings_per_node=max_readings)
    return _buffer_manager


def reset_buffer_manager() -> None:
    """Reset the global buffer manager (useful for testing)."""
    global _buffer_manager
    if _buffer_manager is not None:
        _buffer_manager.clear_all()
    _buffer_manager = None


# ============================================================
# Utility Functions
# ============================================================

def create_features_from_stats(
    stats: RollingStats,
    hour_of_day: Optional[int] = None,
    is_daytime: Optional[int] = None
) -> Dict[str, float]:
    """
    Convert RollingStats to a feature dictionary for ML models.
    This provides the enhanced feature set for inference.
    """
    if hour_of_day is None:
        hour_of_day = datetime.now(timezone.utc).hour
    
    if is_daytime is None:
        is_daytime = 1 if 6 <= hour_of_day < 18 else 0
    
    # Calculate comfort index
    def calc_comfort(temp, humid, nh3):
        temp_score = max(0, min(1, 1 - abs(temp - 28) / 15))
        humid_score = max(0, min(1, 1 - abs(humid - 80) / 35))
        nh3_score = max(0, min(1, 1 - (nh3 / 20)))
        return (temp_score * 0.35 + humid_score * 0.35 + nh3_score * 0.30) * 100
    
    comfort = calc_comfort(stats.latest_temp, stats.latest_humid, stats.latest_ammonia)
    
    return {
        # Base features (compatible with existing models)
        "temperature": stats.latest_temp,
        "humidity": stats.latest_humid,
        "ammonia": stats.latest_ammonia,
        "hour_of_day": hour_of_day,
        "is_daytime": is_daytime,
        "comfort_index": comfort,
        
        # Rolling averages (compatible with existing)
        "temp_avg_1h": stats.temp_avg_60min,
        "humid_avg_1h": stats.humid_avg_60min,
        "nh3_avg_1h": stats.nh3_avg_60min,
        
        # Delta features (compatible with existing)
        "temp_delta_1h": stats.temp_delta_pct_10min,  # Using 10min as proxy
        "humid_delta_1h": stats.humid_delta_pct_10min,
        "nh3_delta_1h": stats.nh3_delta_pct_10min,
        
        # Enhanced features (new)
        "temp_avg_10min": stats.temp_avg_10min,
        "temp_avg_30min": stats.temp_avg_30min,
        "humid_avg_10min": stats.humid_avg_10min,
        "humid_avg_30min": stats.humid_avg_30min,
        "nh3_avg_10min": stats.nh3_avg_10min,
        "nh3_avg_30min": stats.nh3_avg_30min,
        
        "temp_std_30min": stats.temp_std_30min,
        "humid_std_30min": stats.humid_std_30min,
        "nh3_std_30min": stats.nh3_std_30min,
        
        # Trend as numeric (for ML)
        "temp_trend_numeric": 1 if stats.temp_trend == "rising" else (-1 if stats.temp_trend == "falling" else 0),
        "humid_trend_numeric": 1 if stats.humid_trend == "rising" else (-1 if stats.humid_trend == "falling" else 0),
        "nh3_trend_numeric": 1 if stats.nh3_trend == "rising" else (-1 if stats.nh3_trend == "falling" else 0),
    }


# ============================================================
# Example usage / testing
# ============================================================

if __name__ == "__main__":
    import random
    
    print("=== Buffer Manager Test ===\n")
    
    # Get buffer manager
    bm = get_buffer_manager(max_readings=60)
    
    # Simulate sensor readings for 2 nodes
    nodes = ["node-A", "node-B"]
    
    print("Simulating 15 readings per node (every 1 minute)...\n")
    
    base_time = time.time() - 900  # Start 15 minutes ago
    
    for i in range(15):
        for node_id in nodes:
            # Simulate sensor values with some noise
            temp = 28 + random.uniform(-2, 3) + (i * 0.1)  # Slight upward trend
            humid = 78 + random.uniform(-3, 3)
            nh3 = 8 + random.uniform(-1, 2) + (i * 0.05)
            
            timestamp = base_time + (i * 60)  # 1 reading per minute
            
            stats = bm.push_reading(
                node_id=node_id,
                temperature=temp,
                humidity=humid,
                ammonia=nh3,
                timestamp=timestamp
            )
    
    print("=== Stats for node-A ===")
    stats_a = bm.get_stats("node-A")
    for key, value in stats_a.to_dict().items():
        print(f"  {key}: {value}")
    
    print("\n=== Stats for node-B ===")
    stats_b = bm.get_stats("node-B")
    for key, value in stats_b.to_dict().items():
        print(f"  {key}: {value}")
    
    print("\n=== Memory Usage ===")
    mem = bm.memory_usage_estimate()
    for key, value in mem.items():
        print(f"  {key}: {value}")
    
    print("\n=== Features for ML (node-A) ===")
    features = create_features_from_stats(stats_a)
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    print("\n=== Test Complete ===")
