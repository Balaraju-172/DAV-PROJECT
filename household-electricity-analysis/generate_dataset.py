"""
Dataset Generator - Household Electricity Consumption
=====================================================
Generates a realistic sample dataset of household electricity readings.

The data simulates realistic household behaviour:
  - Lower consumption at night (00:00-05:00)
  - Morning ramp-up (06:00-09:00)  - lights, kitchen, water heater
  - Moderate afternoon consumption (10:00-16:00)
  - High evening consumption (17:00-22:00) - family returns home
  - Late-night decline (23:00)
  - Weekday vs weekend differences
  - Seasonal (monthly) variation - summer/winter extremes
  - ~2-5% unusually high-consumption records
  - Physically consistent Voltage, Current and Power (P = V x I)

Run:
    python generate_dataset.py
"""

import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
N_ROWS = 5000
OUTPUT_PATH = os.path.join("data", "electricity_consumption.csv")
RANDOM_SE = 42

np.random.seed(RANDOM_SE)

# ----------------------------------------------------------------------
# 1. Build a realistic date/time base (full year 2026)
# ----------------------------------------------------------------------
start = pd.Timestamp("2026-01-01")
random_day_offsets = np.random.randint(0, 365, size=N_ROWS)
random_hours = np.random.randint(0, 24, size=N_ROWS)
timestamps = start + pd.to_timedelta(random_day_offsets, unit="D") + pd.to_timedelta(random_hours, unit="h")

df = pd.DataFrame({"Timestamp": timestamps})
df["Date"] = df["Timestamp"].dt.strftime("%Y-%m-%d")
df["Time"] = df["Timestamp"].dt.strftime("%H:%M")
df["Day"] = df["Timestamp"].dt.day_name()
df["Hour"] = df["Timestamp"].dt.hour
df["Month_Name"] = df["Timestamp"].dt.strftime("%B")
df["Day_Type"] = np.where(df["Timestamp"].dt.dayofweek >= 5, "Weekend", "Weekday")

# ----------------------------------------------------------------------
# 2. Hour-of-day base load profile (kW) - realistic household shape
# ----------------------------------------------------------------------
hour_base_load = {
    0: 0.35, 1: 0.30, 2: 0.28, 3: 0.27, 4: 0.28, 5: 0.32,
    6: 0.60, 7: 1.10, 8: 1.30, 9: 0.90, 10: 0.75, 11: 0.85,
    12: 1.00, 13: 0.80, 14: 0.70, 15: 0.70, 16: 0.80,
    17: 1.30, 18: 1.70, 19: 2.00, 20: 1.90, 21: 1.50, 22: 0.90, 23: 0.55,
}
df["Base_Load_kW"] = df["Hour"].map(hour_base_load)

# ----------------------------------------------------------------------
# 3. Weekend / seasonal / daily multipliers
# ----------------------------------------------------------------------
# Weekends: later wake-up, more daytime usage, later evenings
weekend_multiplier = {
    0: 0.85, 1: 0.90, 2: 0.90, 3: 0.90, 4: 0.95, 5: 1.05,
    6: 0.70, 7: 0.50, 8: 0.60, 9: 1.10, 10: 1.25, 11: 1.25,
    12: 1.30, 13: 1.25, 14: 1.20, 15: 1.15, 16: 1.15,
    17: 1.20, 18: 1.25, 19: 1.30, 20: 1.25, 21: 1.15, 22: 1.00, 23: 0.90,
}
df["Weekend_Mult"] = np.where(
    df["Day_Type"] == "Weekend",
    df["Hour"].map(weekend_multiplier),
    1.0,  # weekday: base profile already represents weekdays
)

# Seasonal multiplier: fans/AC in summer, heaters in winter, mild spring/autumn
seasonal = {1: 1.25, 2: 1.20, 3: 1.00, 4: 0.95, 5: 0.90, 6: 1.15,
            7: 1.30, 8: 1.25, 9: 1.00, 10: 0.90, 11: 1.05, 12: 1.20}
df["Seasonal_Factor"] = df["Timestamp"].dt.month.map(seasonal)

# Day-to-day household randomness
df["Daily_Factor"] = np.random.normal(1.0, 0.12, size=N_ROWS).clip(0.6, 1.6)

# ----------------------------------------------------------------------
# 4. Final power consumption (kW)
# ----------------------------------------------------------------------
df["Power_Consumption_kW"] = (
    df["Base_Load_kW"] * df["Weekend_Mult"] * df["Seasonal_Factor"] * df["Daily_Factor"]
)
# Small measurement noise
df["Power_Consumption_kW"] *= np.random.normal(1.0, 0.06, size=N_ROWS)
df["Power_Consumption_kW"] = df["Power_Consumption_kW"].clip(0.05, None)

# ----------------------------------------------------------------------
# 5. Voltage and Current (physically consistent: P = V x I)
# ----------------------------------------------------------------------
# Indian household supply is ~230 V nominal with a few volts of variation.
df["Voltage_V"] = np.random.normal(230, 2.5, size=N_ROWS).clip(215, 245)
# Higher load tends to slightly reduce supply voltage (realistic behaviour)
df["Voltage_V"] -= (df["Power_Consumption_kW"] > 1.5) * np.random.uniform(0, 3, size=N_ROWS)

# Current = Power / Voltage  (kW -> W, so A = P*1000 / V)
df["Global_Intensity_A"] = ((df["Power_Consumption_kW"] * 1000) / df["Voltage_V"]).clip(0.05, None)

# ----------------------------------------------------------------------
# 6. Energy for the measurement interval (1-hour interval)
# ----------------------------------------------------------------------
# Each reading represents a 1-hour interval, so numerically kWh = kW x 1h.
df["Energy_Consumption_kWh"] = df["Power_Consumption_kW"]

# ----------------------------------------------------------------------
# 7. Room-wise breakdown (estimated split of total energy)
# ----------------------------------------------------------------------
def room_shares(hour):
    """Return (kitchen, living, bedroom, other) share for a given hour."""
    if 0 <= hour <= 5:        # night: fans/AC in bedrooms dominate
        return 0.08, 0.12, 0.55, 0.25
    elif 6 <= hour <= 9:      # morning: kitchen (breakfast, water heater)
        return 0.42, 0.20, 0.13, 0.25
    elif 10 <= hour <= 16:    # afternoon: mixed, TV/living room
        return 0.30, 0.34, 0.13, 0.23
    else:                     # evening: living room (TV, lights) + kitchen (dinner)
        return 0.34, 0.33, 0.13, 0.20

shares = np.array(df["Hour"].apply(room_shares).tolist())
df["Kitchen_Usage_kWh"] = shares[:, 0] * df["Energy_Consumption_kWh"]
df["Living_Room_Usage_kWh"] = shares[:, 1] * df["Energy_Consumption_kWh"]
df["Bedroom_Usage_kWh"] = shares[:, 2] * df["Energy_Consumption_kWh"]
df["Other_Usage_kWh"] = shares[:, 3] * df["Energy_Consumption_kWh"]

# ----------------------------------------------------------------------
# 8. Unusually high-consumption records (~3%)
# ----------------------------------------------------------------------
# Simulates forgotten appliances left running, guests visiting, parties, etc.
unusual_mask = np.random.rand(N_ROWS) < 0.03
spike_multiplier = np.random.uniform(1.8, 2.6, size=N_ROWS)
df.loc[unusual_mask, "Power_Consumption_kW"] *= spike_multiplier[unusual_mask]
df.loc[unusual_mask, "Energy_Consumption_kWh"] = df.loc[unusual_mask, "Power_Consumption_kW"]

# Scale the room breakdown proportionally so totals stay consistent,
# then recompute current so P = V x I still holds.
for col in ["Kitchen_Usage_kWh", "Living_Room_Usage_kWh", "Bedroom_Usage_kWh", "Other_Usage_kWh"]:
    df.loc[unusual_mask, col] *= spike_multiplier[unusual_mask]

df["Global_Intensity_A"] = ((df["Power_Consumption_kW"] * 1000) / df["Voltage_V"]).clip(0.05, None)

# ----------------------------------------------------------------------
# 9. Assemble final dataset
# ----------------------------------------------------------------------
final = pd.DataFrame({
    "Record_ID": np.arange(1, N_ROWS + 1),
    "Date": df["Date"],
    "Time": df["Time"],
    "Day": df["Day"],
    "Day_Type": df["Day_Type"],
    "Month": df["Month_Name"],
    "Hour": df["Hour"],
    "Voltage_V": df["Voltage_V"].round(2),
    "Global_Intensity_A": df["Global_Intensity_A"].round(3),
    "Power_Consumption_kW": df["Power_Consumption_kW"].round(4),
    "Energy_Consumption_kWh": df["Energy_Consumption_kWh"].round(4),
    "Kitchen_Usage_kWh": df["Kitchen_Usage_kWh"].round(4),
    "Living_Room_Usage_kWh": df["Living_Room_Usage_kWh"].round(4),
    "Bedroom_Usage_kWh": df["Bedroom_Usage_kWh"].round(4),
    "Other_Usage_kWh": df["Other_Usage_kWh"].round(4),
})

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
final.to_csv(OUTPUT_PATH, index=False)

print("Dataset generated successfully!")
print(f"Rows: {len(final)}")
print(f"Columns: {len(final.columns)}")
print(f"File: {OUTPUT_PATH}")
