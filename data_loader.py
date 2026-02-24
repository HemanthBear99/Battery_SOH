"""
Data Loader Module for Battery SoH Estimation
Handles loading and preprocessing of NASA and CALCE datasets SEPARATELY.

NASA: Multi-cycle degradation data, split by battery.
CALCE: Single-cycle profiles, use pre-existing Train/Test folder split.
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, Optional
from feature_engineering import BatteryFeatureEngineer

logger = logging.getLogger(__name__)

# Fixed sequence lengths for all datasets — ensures consistent batching
SEQ_LEN = 500   # Voltage curve points
IC_LEN = 200    # IC curve points (matches BatteryFeatureEngineer.num_points)


class BatteryDataset(Dataset):
    """Custom Dataset for Battery SoH estimation."""

    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # All fields are guaranteed non-None with fixed shapes after pipeline changes
        return {
            "battery_id": sample["battery_id"],
            "cycle_index": torch.tensor(sample["cycle_index"], dtype=torch.float32),
            "target_soh": torch.tensor(sample["soh"], dtype=torch.float32),
            "voltage_curve": torch.tensor(sample["voltage_curve"], dtype=torch.float32),
            "ic_curve": torch.tensor(sample["ic_curve"], dtype=torch.float32),
            "graph_features": torch.tensor(sample["graph_features"], dtype=torch.float32),
        }


class DataLoaderFactory:
    """
    Factory class for creating data loaders.
    Handles NASA and CALCE datasets SEPARATELY — never combines them.
    """

    def __init__(self, dataset_path: str):
        if not os.path.isabs(dataset_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.dataset_path = os.path.join(base_dir, dataset_path)
        else:
            self.dataset_path = dataset_path

        self.nasa_path = os.path.join(self.dataset_path, "Nasa dataset")
        self.calce_path = os.path.join(self.dataset_path, "CALCE dataset")

    # ================================================================
    # NASA DATASET
    # ================================================================

    def load_nasa_dataset(self) -> pd.DataFrame:
        """
        Load NASA dataset: discharge cycles with capacity degradation.

        Returns:
            DataFrame with battery_id, cycle_index, capacity, temperature,
            voltage_curve, current_curve, time_curve
        """
        metadata_path = os.path.join(self.nasa_path, "metadata.csv")
        data_path = os.path.join(self.nasa_path, "data")

        metadata = pd.read_csv(metadata_path)
        metadata["Capacity"] = pd.to_numeric(metadata["Capacity"], errors="coerce")
        metadata["Re"] = pd.to_numeric(metadata["Re"], errors="coerce")
        metadata["Rct"] = pd.to_numeric(metadata["Rct"], errors="coerce")

        discharge_cycles = metadata[metadata["type"] == "discharge"].copy()
        discharge_cycles = discharge_cycles.dropna(subset=["Capacity"])
        discharge_cycles = discharge_cycles[discharge_cycles["Capacity"] > 0]

        processed_data = []

        for _, row in discharge_cycles.iterrows():
            battery_id = row["battery_id"]
            cycle_idx = row["test_id"]
            capacity = float(row["Capacity"])
            temp = row["ambient_temperature"]
            filename = row["filename"]

            file_path = os.path.join(data_path, filename)
            if not os.path.exists(file_path):
                continue

            try:
                cycle_data = pd.read_csv(file_path)

                if "Voltage_measured" not in cycle_data.columns:
                    continue

                voltage_curve = cycle_data["Voltage_measured"].values[:500]
                current_curve = cycle_data["Current_measured"].values[:500]
                time_curve = cycle_data["Time"].values[:500]

                if len(voltage_curve) < 50:
                    continue

                processed_data.append(
                    {
                        "battery_id": battery_id,
                        "cycle_index": int(cycle_idx),
                        "capacity": capacity,
                        "temperature": temp,
                        "voltage_curve": voltage_curve,
                        "current_curve": current_curve,
                        "time_curve": time_curve,
                    }
                )
            except (ValueError, KeyError, pd.errors.ParserError) as e:
                logger.warning(f"Skipping NASA file {filename}: {e}")
                continue

        logger.info(f"NASA: loaded {len(processed_data)} discharge cycles")
        return pd.DataFrame(processed_data)

    # ================================================================
    # CALCE DATASET
    # ================================================================

    def load_calce_split(self, split: str) -> pd.DataFrame:
        """
        Load one split of the CALCE dataset (Train or Test).

        Each CSV file = one discharge cycle under a specific driving profile.
        Columns: Data_Point, Test_Time_s_, Step_Time_s_, Step_Index, Cycle_Index,
                 I, V, ChargeCapacityAh, Discharge_CapacityAh, ..., T, SOC, QAccu

        Args:
            split: 'Train' or 'Test'

        Returns:
            DataFrame with one row per sliding window extracted from each file.
        """
        split_path = os.path.join(self.calce_path, split)
        if not os.path.exists(split_path):
            logger.warning(f"CALCE {split} path not found: {split_path}")
            return pd.DataFrame()

        processed_data = []

        for filename in sorted(os.listdir(split_path)):
            if not filename.endswith(".csv"):
                continue

            file_path = os.path.join(split_path, filename)

            try:
                df = pd.read_csv(file_path)

                if "V" not in df.columns or "I" not in df.columns:
                    continue

                # Parse filename: e.g. TBJDST_4580.csv -> profile=TBJDST, temp=45, dod=80
                parts = filename.replace(".csv", "").split("_")
                profile = parts[0]
                condition = parts[1] if len(parts) > 1 else "0000"

                # Temperature is encoded in filename, NOT in the T column (which is 0)
                if len(condition) == 4:
                    temp = int(condition[:2])
                    dod = int(condition[2:])
                elif len(condition) == 3:
                    temp = int(condition[0])
                    dod = int(condition[1:])
                else:
                    temp = 25
                    dod = 80

                # Max discharge capacity = the SoH-relevant metric for this file
                max_discharge_cap = df["Discharge_CapacityAh"].max()

                # Get the active discharge portion (Step_Index 7 = dynamic driving)
                active = df[df["Step_Index"] == 7] if "Step_Index" in df.columns else df
                if len(active) < 100:
                    active = df

                voltage = active["V"].values
                current = active["I"].values
                time_vals = active["Test_Time_s_"].values

                if len(voltage) < 100:
                    continue

                # Battery ID from filename
                battery_id = f"CALCE_{profile}_{condition}"

                # Generate sliding windows of 500 points (stride=250) for more samples
                window_size = 500
                stride = 250

                for start in range(0, len(voltage) - window_size + 1, stride):
                    end = start + window_size

                    v_window = voltage[start:end]
                    c_window = current[start:end]
                    t_window = time_vals[start:end]

                    # SOC at the midpoint of this window as a proxy for "local SoH"
                    if "SOC" in active.columns:
                        mid_idx = start + window_size // 2
                        soc_at_mid = active["SOC"].values[mid_idx]
                    else:
                        soc_at_mid = 1.0 - (start / max(len(voltage) - 1, 1))

                    processed_data.append(
                        {
                            "battery_id": battery_id,
                            "cycle_index": start // stride,
                            "capacity": max_discharge_cap,
                            "temperature": temp,
                            "dod": dod,
                            "profile": profile,
                            "voltage_curve": v_window.astype(np.float32),
                            "current_curve": c_window.astype(np.float32),
                            "time_curve": t_window.astype(np.float64),
                            "soc_at_mid": soc_at_mid,
                        }
                    )

                # Also add full file as one sample (truncated/padded to 500)
                full_v = voltage[:500]
                full_c = current[:500]
                full_t = time_vals[:500]

                processed_data.append(
                    {
                        "battery_id": battery_id,
                        "cycle_index": 0,
                        "capacity": max_discharge_cap,
                        "temperature": temp,
                        "dod": dod,
                        "profile": profile,
                        "voltage_curve": full_v.astype(np.float32),
                        "current_curve": full_c.astype(np.float32),
                        "time_curve": full_t.astype(np.float64),
                        "soc_at_mid": active["SOC"].values[min(250, len(active) - 1)]
                        if "SOC" in active.columns
                        else 0.5,
                    }
                )

            except (ValueError, KeyError, pd.errors.ParserError) as e:
                logger.warning(f"Error loading CALCE {filename}: {e}")
                continue

        logger.info(f"CALCE {split}: loaded {len(processed_data)} samples from {split_path}")
        return pd.DataFrame(processed_data)

    # ================================================================
    # MAIN ENTRY POINT
    # ================================================================

    def create_dataloaders(
        self,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        seed: int = 42,
        dataset: str = "nasa",
    ) -> Tuple[DataLoader, DataLoader, List[str]]:
        """
        Create train and test data loaders.

        NASA: split by battery (train_ratio).
        CALCE: uses pre-existing Train/ and Test/ folders.

        NEVER combines datasets.

        Args:
            batch_size: Batch size
            train_ratio: Train ratio (NASA only; CALCE uses folder split)
            seed: Random seed
            dataset: 'nasa' or 'calce' — required

        Returns:
            train_loader, test_loader, test_battery_ids
        """
        np.random.seed(seed)

        if dataset.lower() == "nasa":
            return self._create_nasa_loaders(batch_size, train_ratio, seed)
        elif dataset.lower() == "calce":
            return self._create_calce_loaders(batch_size)
        else:
            raise ValueError(f"Unknown dataset: {dataset}. Must be 'nasa' or 'calce'.")

    def _create_nasa_loaders(
        self, batch_size: int, train_ratio: float, seed: int
    ) -> Tuple[DataLoader, DataLoader, List[str]]:
        """
        Create NASA dataloaders with cross-battery split.
        Test set only includes batteries with sufficient degradation signal
        (≥30 cycles AND SoH range ≥0.15) for meaningful R² evaluation.
        """
        nasa_df = self.load_nasa_dataset()

        if len(nasa_df) == 0:
            raise ValueError("NASA dataset is empty. Check the dataset path.")

        print(f"NASA dataset: {len(nasa_df)} discharge cycles from "
              f"{nasa_df['battery_id'].nunique()} batteries")

        # Score batteries by degradation range for proper test selection
        battery_stats = {}
        for bat in nasa_df["battery_id"].unique():
            bat_df = nasa_df[nasa_df["battery_id"] == bat]
            caps = bat_df["capacity"].values
            max_cap = caps.max()
            soh = caps / max_cap
            battery_stats[bat] = {
                "n_cycles": len(bat_df),
                "soh_range": float(soh.max() - soh.min()),
                "soh_std": float(soh.std()),
            }

        # Eligible for test: enough cycles AND enough degradation
        eligible_test = [
            b for b, s in battery_stats.items()
            if s["n_cycles"] >= 30 and s["soh_range"] >= 0.15
        ]
        ineligible = [
            b for b in battery_stats if b not in eligible_test
        ]

        np.random.seed(seed)
        np.random.shuffle(eligible_test)

        # Take ~20% of eligible batteries for test
        n_test = max(int(len(eligible_test) * (1 - train_ratio)), 3)
        test_batteries = eligible_test[:n_test]
        train_batteries = eligible_test[n_test:] + ineligible  # rest + ineligible all train

        print(f"\nBattery selection:")
        print(f"  Eligible for test ({len(eligible_test)}): >=30 cycles AND SoH range >=0.15")
        print(f"  Train batteries ({len(train_batteries)}): {sorted(train_batteries)}")
        print(f"  Test batteries ({len(test_batteries)}): {sorted(test_batteries)}")

        train_df = nasa_df[nasa_df["battery_id"].isin(train_batteries)]
        test_df = nasa_df[nasa_df["battery_id"].isin(test_batteries)]

        train_data = self._prepare_nasa_samples(train_df)
        test_data = self._prepare_nasa_samples(test_df)

        print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

        train_loader = DataLoader(
            BatteryDataset(train_data),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_graph_samples,
        )
        test_loader = DataLoader(
            BatteryDataset(test_data),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_graph_samples,
        )

        return train_loader, test_loader, list(test_batteries)

    def _create_calce_loaders(
        self, batch_size: int
    ) -> Tuple[DataLoader, DataLoader, List[str]]:
        """Create CALCE dataloaders using pre-existing Train/Test folders."""
        train_df = self.load_calce_split("Train")
        test_df = self.load_calce_split("Test")

        if len(train_df) == 0:
            raise ValueError("CALCE Train folder is empty.")
        if len(test_df) == 0:
            raise ValueError("CALCE Test folder is empty.")

        print(f"CALCE Train: {len(train_df)} samples from "
              f"{train_df['battery_id'].nunique()} profiles")
        print(f"CALCE Test: {len(test_df)} samples from "
              f"{test_df['battery_id'].nunique()} profiles")

        # Reference capacity = max capacity across all training files
        ref_capacity = train_df["capacity"].max()
        print(f"Reference capacity: {ref_capacity:.4f} Ah")

        train_data = self._prepare_calce_samples(train_df, ref_capacity)
        test_data = self._prepare_calce_samples(test_df, ref_capacity)

        print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

        test_batteries = list(test_df["battery_id"].unique())

        train_loader = DataLoader(
            BatteryDataset(train_data),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_graph_samples,
        )
        test_loader = DataLoader(
            BatteryDataset(test_data),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_graph_samples,
        )

        return train_loader, test_loader, test_batteries

    # ================================================================
    # SAMPLE PREPARATION
    # ================================================================

    def _prepare_nasa_samples(self, df: pd.DataFrame) -> List[Dict]:
        """
        Prepare NASA samples.

        SoH = Capacity / Max_Capacity_for_that_battery (degradation tracking).
        """
        samples = []
        fe = BatteryFeatureEngineer()

        for battery_id in df["battery_id"].unique():
            battery_df = df[df["battery_id"] == battery_id].sort_values("cycle_index")

            capacities = battery_df["capacity"].values
            max_capacity = float(capacities.max())
            if max_capacity <= 0 or np.isnan(max_capacity):
                continue

            cycle_indices = battery_df["cycle_index"].values.astype(float)
            max_cycle = max(cycle_indices.max(), 1.0)

            for _, row in battery_df.iterrows():
                capacity_val = float(row["capacity"])
                soh = capacity_val / max_capacity

                # FIXED-BOUND voltage normalization: preserves absolute voltage level
                # A degraded battery has LOWER voltage — per-sample norm erases this!
                V_MIN, V_MAX = 2.0, 4.5  # Li-ion operating range
                mean_voltage = 0.5  # default
                if row["voltage_curve"] is not None:
                    vc = np.array(row["voltage_curve"], dtype=np.float32)
                    mean_voltage = float(vc.mean() - V_MIN) / (V_MAX - V_MIN)
                    vc_norm = np.clip((vc - V_MIN) / (V_MAX - V_MIN), 0, 1)
                    # Pad/truncate to fixed SEQ_LEN for consistent batching
                    if len(vc_norm) < SEQ_LEN:
                        vc_norm = np.pad(vc_norm, (0, SEQ_LEN - len(vc_norm)))
                    voltage_curve = vc_norm[:SEQ_LEN]
                else:
                    voltage_curve = np.zeros(SEQ_LEN, dtype=np.float32)

                # Compute IC curve — fallback to zeros so every sample has one
                ic_curve = np.zeros(IC_LEN, dtype=np.float32)
                ic_features = {}
                if row["voltage_curve"] is not None and row["current_curve"] is not None:
                    raw_v = np.array(row["voltage_curve"], dtype=np.float64)
                    raw_c = np.array(row["current_curve"], dtype=np.float64)
                    raw_t = np.array(row["time_curve"], dtype=np.float64) if row["time_curve"] is not None else np.arange(len(raw_v)) * 10.0

                    ic_raw, v_pts = fe.compute_incremental_capacity(raw_v, raw_c, raw_t)
                    if len(ic_raw) >= IC_LEN:
                        ic_min, ic_max = ic_raw.min(), ic_raw.max()
                        if ic_max > ic_min:
                            ic_curve = ((ic_raw - ic_min) / (ic_max - ic_min)).astype(np.float32)[:IC_LEN]
                        else:
                            ic_curve = np.zeros(IC_LEN, dtype=np.float32)
                        ic_features = fe.extract_ic_features(ic_raw, v_pts)

                # Normalized cycle index
                norm_cycle = float(row["cycle_index"]) / max_cycle

                # Graph features: 8 nodes, 1 feature each — NO target leakage
                temp = float(row.get("temperature", 25)) / 100.0
                graph_node_values = [
                    norm_cycle,
                    temp,
                    float(ic_features.get("ic_peak_height", 0.0)),
                    float(ic_features.get("ic_peak_location", 3.7)) / 4.2,
                    float(ic_features.get("ic_peak_width", 0.0)),
                    float(ic_features.get("ic_area", 0.0)),
                    float(ic_features.get("ic_num_peaks", 0)) / 5.0,
                    mean_voltage,  # KEY: average voltage level = health indicator
                ]
                graph_features = np.array(graph_node_values, dtype=np.float32).reshape(-1, 1)

                samples.append(
                    {
                        "battery_id": row["battery_id"],
                        "cycle_index": norm_cycle,
                        "soh": np.clip(soh, 0, 1.5),
                        "voltage_curve": voltage_curve,
                        "ic_curve": ic_curve,
                        "graph_features": graph_features,
                    }
                )

        return samples

    def _prepare_calce_samples(self, df: pd.DataFrame, ref_capacity: float) -> List[Dict]:
        """
        Prepare CALCE samples.

        SoH = Discharge_Capacity / Reference_Capacity.
        Each sliding window becomes one sample.
        """
        samples = []
        fe = BatteryFeatureEngineer()

        for _, row in df.iterrows():
            capacity_val = float(row["capacity"])
            soh = capacity_val / ref_capacity if ref_capacity > 0 else 1.0

            # FIXED-BOUND normalization — preserves absolute voltage level (= health signal).
            # Per-sample min-max normalization was removing the degradation information!
            V_MIN_CALCE, V_MAX_CALCE = 2.5, 4.2  # Li-ion EV profile operating range
            if row["voltage_curve"] is not None:
                vc = np.array(row["voltage_curve"], dtype=np.float32)
                vc_norm = np.clip((vc - V_MIN_CALCE) / (V_MAX_CALCE - V_MIN_CALCE), 0, 1)
                # Pad/truncate to fixed SEQ_LEN for consistent batching
                if len(vc_norm) < SEQ_LEN:
                    vc_norm = np.pad(vc_norm, (0, SEQ_LEN - len(vc_norm)))
                voltage_curve = vc_norm[:SEQ_LEN]
            else:
                voltage_curve = np.zeros(SEQ_LEN, dtype=np.float32)

            # Compute IC curve — fallback to zeros so every sample has one
            ic_curve = np.zeros(IC_LEN, dtype=np.float32)
            ic_features = {}
            if row["voltage_curve"] is not None and row["current_curve"] is not None:
                raw_v = np.array(row["voltage_curve"], dtype=np.float64)
                raw_c = np.array(row["current_curve"], dtype=np.float64)
                raw_t = np.array(row["time_curve"], dtype=np.float64) if row.get("time_curve") is not None else np.arange(len(raw_v)) * 10.0

                ic_raw, v_pts = fe.compute_incremental_capacity(raw_v, raw_c, raw_t)
                if len(ic_raw) >= IC_LEN:
                    ic_min, ic_max = ic_raw.min(), ic_raw.max()
                    if ic_max > ic_min:
                        ic_curve = ((ic_raw - ic_min) / (ic_max - ic_min)).astype(np.float32)[:IC_LEN]
                    else:
                        ic_curve = np.zeros(IC_LEN, dtype=np.float32)
                    ic_features = fe.extract_ic_features(ic_raw, v_pts)

            norm_cycle = float(row["cycle_index"]) / max(df["cycle_index"].max(), 1)

            temp = float(row.get("temperature", 25)) / 100.0
            dod = float(row.get("dod", 80)) / 100.0

            graph_node_values = [
                norm_cycle,
                temp,
                float(ic_features.get("ic_peak_height", 0.0)),
                float(ic_features.get("ic_peak_location", 3.7)) / 4.2,
                float(ic_features.get("ic_peak_width", 0.0)),
                float(ic_features.get("ic_area", 0.0)),
                float(ic_features.get("ic_num_peaks", 0)) / 5.0,
                dod,
            ]
            graph_features = np.array(graph_node_values, dtype=np.float32).reshape(-1, 1)

            samples.append(
                {
                    "battery_id": row["battery_id"],
                    "cycle_index": norm_cycle,
                    "soh": np.clip(soh, 0, 1.5),
                    "voltage_curve": voltage_curve,
                    "ic_curve": ic_curve,
                    "graph_features": graph_features,
                }
            )

        return samples


def collate_graph_samples(batch):
    """
    Collate function for batches with graph data.

    All sequences are fixed-length (voltage=SEQ_LEN, IC=IC_LEN) so no
    variable-length padding is needed — just stack tensors directly.
    """
    battery_ids = [sample["battery_id"] for sample in batch]
    cycle_indices = torch.stack([sample["cycle_index"] for sample in batch])
    target_soh = torch.stack([sample["target_soh"] for sample in batch])

    # All sequences are guaranteed fixed-length — stack directly
    voltage_curves = torch.stack([sample["voltage_curve"] for sample in batch])
    ic_curves = torch.stack([sample["ic_curve"] for sample in batch])
    graph_features = torch.stack([sample["graph_features"] for sample in batch])

    return {
        "battery_ids": battery_ids,
        "cycle_indices": cycle_indices,
        "target_soh": target_soh,
        "voltage_curves": voltage_curves,
        "ic_curves": ic_curves,
        "graph_features": graph_features,
    }
