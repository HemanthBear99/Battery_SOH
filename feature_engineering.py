"""
Feature Engineering Module for Battery SoH Estimation
Computes Incremental Capacity (IC) curves, resistance features, and temperature features
"""

import logging
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger(__name__)


class BatteryFeatureEngineer:
    """
    Feature engineering for battery SoH estimation
    Computes IC curves, extracts features, normalizes data
    """

    def __init__(
        self,
        voltage_smoothing: bool = True,
        ic_smoothing: bool = True,
        min_voltage: float = 2.0,
        max_voltage: float = 4.2,
        num_points: int = 200,
    ):
        self.voltage_smoothing = voltage_smoothing
        self.ic_smoothing = ic_smoothing
        self.min_voltage = min_voltage
        self.max_voltage = max_voltage
        self.num_points = num_points

    def compute_incremental_capacity(
        self, voltage: np.ndarray, current: np.ndarray, time: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Incremental Capacity (IC) curve

        IC = dQ/dV where Q is capacity (Ah) and V is voltage (V)

        dQ = I * dt

        Args:
            voltage: Voltage measurements (V)
            current: Current measurements (A)
            time: Time measurements (s)

        Returns:
            ic_curve, voltage_points
        """
        if len(voltage) < 10:
            return np.array([]), np.array([])

        try:
            dt = np.diff(time)
            dt = np.append(dt, dt[-1] if len(dt) > 0 else 0)

            dQ = current * dt / 3600

            cumulative_Q = np.cumsum(dQ)
            cumulative_Q = cumulative_Q - cumulative_Q[0]

            if self.voltage_smoothing and len(voltage) > 11:
                voltage = savgol_filter(voltage, window_length=11, polyorder=3)

            valid_mask = (voltage > self.min_voltage) & (voltage < self.max_voltage)
            if not valid_mask.any():
                return np.array([]), np.array([])

            voltage_valid = voltage[valid_mask]
            cumulative_Q_valid = cumulative_Q[valid_mask]

            if len(voltage_valid) < 10:
                return np.array([]), np.array([])

            # Sort by voltage — interp1d requires monotonically increasing x
            sort_idx = np.argsort(voltage_valid)
            voltage_sorted = voltage_valid[sort_idx]
            Q_sorted = cumulative_Q_valid[sort_idx]

            # Remove duplicate voltage points (cause interp1d to fail)
            unique_mask = np.diff(voltage_sorted, prepend=-np.inf) > 1e-6
            voltage_sorted = voltage_sorted[unique_mask]
            Q_sorted = Q_sorted[unique_mask]

            if len(voltage_sorted) < 5:
                return np.array([]), np.array([])

            voltage_grid = np.linspace(
                voltage_sorted.min() + 0.05, voltage_sorted.max() - 0.05, self.num_points
            )

            # Bounded fill: flat extrapolation prevents wild values outside voltage range
            interp_func = interp1d(
                voltage_sorted,
                Q_sorted,
                kind="linear",
                bounds_error=False,
                fill_value=(Q_sorted[0], Q_sorted[-1]),
            )

            Q_interp = interp_func(voltage_grid)

            dQ_dV = np.gradient(Q_interp, voltage_grid)

            if self.ic_smoothing and len(dQ_dV) > 11:
                dQ_dV = savgol_filter(dQ_dV, window_length=11, polyorder=3)

            return dQ_dV, voltage_grid

        except (ValueError, np.linalg.LinAlgError) as e:
            logger.debug(f"IC curve computation failed: {e}")
            return np.array([]), np.array([])

    def extract_ic_features(
        self, ic_curve: np.ndarray, voltage_points: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract features from IC curve

        Features:
        - IC peak height
        - IC peak location (voltage)
        - IC peak width
        - IC area under curve

        Args:
            ic_curve: IC curve values
            voltage_points: Corresponding voltage points

        Returns:
            Dictionary of IC features
        """
        features = {
            "ic_peak_height": 0.0,
            "ic_peak_location": 3.7,
            "ic_peak_width": 0.0,
            "ic_area": 0.0,
            "ic_num_peaks": 0,
        }

        if len(ic_curve) < 5:
            return features

        try:
            min_ic = np.min(ic_curve)
            max_ic = np.max(ic_curve)
            ic_range = max_ic - min_ic

            if ic_range < 0.01:
                return features

            peaks, peak_props = find_peaks(
                ic_curve,
                height=min_ic + 0.3 * ic_range,
                prominence=0.1 * ic_range,
                distance=10,
            )

            features["ic_num_peaks"] = len(peaks)

            if len(peaks) > 0:
                main_peak_idx = peaks[np.argmax(ic_curve[peaks])]

                features["ic_peak_height"] = float(ic_curve[main_peak_idx])
                features["ic_peak_location"] = float(voltage_points[main_peak_idx])

                half_max = (min_ic + max_ic) / 2
                above_half = np.where(ic_curve > half_max)[0]
                if len(above_half) > 0:
                    features["ic_peak_width"] = float(
                        voltage_points[min(above_half[-1], len(voltage_points) - 1)]
                        - voltage_points[above_half[0]]
                    )

            features["ic_area"] = float(np.trapz(ic_curve, voltage_points))

        except (ValueError, IndexError) as e:
            logger.debug(f"IC feature extraction failed: {e}")

        return features

    def compute_resistance_features(
        self,
        Re: Optional[float] = None,
        Rct: Optional[float] = None,
        internal_resistance: Optional[float] = None,
        voltage: Optional[np.ndarray] = None,
        current: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute resistance-related features

        Args:
            Re: Electrolyte resistance (Ohm)
            Rct: Charge transfer resistance (Ohm)
            internal_resistance: Internal resistance (Ohm)
            voltage: Voltage curve
            current: Current curve

        Returns:
            Dictionary of resistance features
        """
        features = {"resistance": 0.1, "Re": 0.0, "Rct": 0.0, "dVdt": 0.0}

        if Re is not None and not np.isnan(Re):
            features["Re"] = float(Re)
            features["resistance"] += float(Re)

        if Rct is not None and not np.isnan(Rct):
            features["Rct"] = float(Rct)
            features["resistance"] += float(Rct)

        if internal_resistance is not None and not np.isnan(internal_resistance):
            features["resistance"] = float(internal_resistance)

        if voltage is not None and current is not None and len(voltage) > 1:
            try:
                dv_dt = np.gradient(voltage, np.arange(len(voltage)))
                features["dVdt"] = float(np.mean(np.abs(dv_dt)))
            except (ValueError, FloatingPointError) as e:
                logger.debug(f"dV/dt computation failed: {e}")

        return features

    def process_cycle(
        self,
        voltage: np.ndarray,
        current: np.ndarray,
        time: np.ndarray,
        temperature: float,
        Re: Optional[float] = None,
        Rct: Optional[float] = None,
        capacity: Optional[float] = None,
        internal_resistance: Optional[float] = None,
    ) -> Dict:
        """
        Process a single cycle and extract all features

        Args:
            voltage: Voltage measurements
            current: Current measurements
            time: Time measurements
            temperature: Ambient/battery temperature
            Re: Electrolyte resistance
            Rct: Charge transfer resistance
            capacity: Battery capacity
            internal_resistance: Internal resistance

        Returns:
            Dictionary with all features
        """
        ic_curve, voltage_points = self.compute_incremental_capacity(
            voltage, current, time
        )

        ic_features = self.extract_ic_features(ic_curve, voltage_points)

        resistance_features = self.compute_resistance_features(
            Re=Re,
            Rct=Rct,
            internal_resistance=internal_resistance,
            voltage=voltage,
            current=current,
        )

        result = {
            "ic_curve": ic_curve,
            "voltage_curve": voltage,
            "voltage_points": voltage_points,
            "temperature": temperature,
            "capacity": capacity,
            **ic_features,
            **resistance_features,
        }

        return result

    def normalize_features(self, features: List[Dict]) -> List[Dict]:
        """
        Normalize all features to [0, 1] range

        Args:
            features: List of feature dictionaries

        Returns:
            Normalized feature dictionaries
        """
        if len(features) == 0:
            return features

        all_values = {}

        for key in features[0].keys():
            if key in ["ic_curve", "voltage_curve", "voltage_points"]:
                continue

            values = [f.get(key, 0) for f in features if f.get(key) is not None]
            if values:
                all_values[key] = {
                    "min": np.min(values),
                    "max": np.max(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                }

        normalized = []
        for feat in features:
            norm_feat = feat.copy()

            for key, stats in all_values.items():
                if key in norm_feat and norm_feat[key] is not None:
                    val = norm_feat[key]
                    if stats["max"] > stats["min"]:
                        norm_feat[key] = (val - stats["min"]) / (
                            stats["max"] - stats["min"]
                        )
                    else:
                        norm_feat[key] = 0.5

            normalized.append(norm_feat)

        return normalized

    def get_feature_vector(self, features: Dict) -> np.ndarray:
        """
        Convert feature dictionary to flat numpy array

        Args:
            features: Feature dictionary

        Returns:
            Flat feature vector
        """
        feature_keys = [
            "ic_peak_height",
            "ic_peak_location",
            "ic_peak_width",
            "ic_area",
            "temperature",
            "resistance",
            "Re",
            "Rct",
            "dVdt",
            "capacity",
        ]

        vector = []
        for key in feature_keys:
            val = features.get(key, 0.0)
            if val is None or np.isnan(val):
                val = 0.0
            vector.append(val)

        return np.array(vector)


def compute_ic_curve(
    voltage: np.ndarray, current: np.ndarray, time: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standalone function to compute IC curve

    Args:
        voltage: Voltage measurements (V)
        current: Current measurements (A)
        time: Time measurements (s)

    Returns:
        IC curve and voltage points
    """
    engineer = BatteryFeatureEngineer()
    return engineer.compute_incremental_capacity(voltage, current, time)


def extract_features_from_cycle(
    voltage: np.ndarray,
    current: np.ndarray,
    time: np.ndarray,
    temperature: float,
    **kwargs,
) -> Dict:
    """
    Standalone function to extract all features from a cycle

    Args:
        voltage: Voltage measurements
        current: Current measurements
        time: Time measurements
        temperature: Temperature
        **kwargs: Additional features (Re, Rct, capacity, etc.)

    Returns:
        Feature dictionary
    """
    engineer = BatteryFeatureEngineer()
    return engineer.process_cycle(voltage, current, time, temperature, **kwargs)
