"""
Analysis Module - Household Electricity Consumption
===================================================
Reusable analysis functions used by the Streamlit dashboard (app.py)
and the Jupyter notebook (notebooks/electricity_analysis.ipynb).

Every function takes a cleaned pandas DataFrame with the standard
columns of the electricity dataset and returns DataFrames / scalars
ready for plotting or display.
"""

import numpy as np
import pandas as pd

# Canonical column names used throughout the project
ENERGY_COL = "Energy_Consumption_kWh"
POWER_COL = "Power_Consumption_kW"
VOLTAGE_COL = "Voltage_V"
CURRENT_COL = "Global_Intensity_A"
DATE_COL = "Date"
HOUR_COL = "Hour"
DAY_TYPE_COL = "Day_Type"
DAY_COL = "Day"
MONTH_COL = "Month"

MONTH_ORDER = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ======================================================================
# Data loading & cleaning
# ======================================================================

def load_data(path="data/electricity_consumption.csv"):
    """Load the raw CSV and return a cleaned DataFrame."""
    df = pd.read_csv(path)
    df = clean_data(df)
    return df


def clean_data(df):
    """Clean and validate the dataset.

    Steps:
      1. Parse the Date column as datetime.
      2. Fill the few missing numeric values with the median
         (median is robust to the high-consumption outliers).
      3. Drop exact duplicate rows and duplicated Record_IDs.
      4. Validate ranges: non-negative power/energy, valid hours,
         plausible household voltage (200-250 V).
    """
    df = df.copy()

    # 1. Date -> datetime
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL])

    # 2. Missing values in numeric columns -> median of that column
    numeric_cols = [VOLTAGE_COL, CURRENT_COL, POWER_COL, ENERGY_COL,
                    "Kitchen_Usage_kWh", "Living_Room_Usage_kWh",
                    "Bedroom_Usage_kWh", "Other_Usage_kWh"]
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # 3. Duplicates
    df = df.drop_duplicates()
    if "Record_ID" in df.columns:
        df = df.drop_duplicates(subset="Record_ID", keep="first")

    # 4. Range validation
    df = df[df[POWER_COL] >= 0]
    df = df[df[ENERGY_COL] >= 0]
    df = df[(df[HOUR_COL] >= 0) & (df[HOUR_COL] <= 23)]
    df = df[(df[VOLTAGE_COL] >= 200) & (df[VOLTAGE_COL] <= 250)]

    df = df.reset_index(drop=True)
    return df


# ======================================================================
# Basic statistics
# ======================================================================

def calculate_statistics(df):
    """Return a dict of headline statistics for the dashboard KPIs."""
    stats = {
        "total_kwh": df[ENERGY_COL].sum(),
        "avg_kwh": df[ENERGY_COL].mean(),
        "min_kwh": df[ENERGY_COL].min(),
        "max_kwh": df[ENERGY_COL].max(),
        "median_kwh": df[ENERGY_COL].median(),
        "std_kwh": df[ENERGY_COL].std(),
        "avg_voltage": df[VOLTAGE_COL].mean(),
        "avg_current": df[CURRENT_COL].mean(),
        "n_records": len(df),
    }

    # Peak / lowest consumption hours
    hourly = df.groupby(HOUR_COL)[POWER_COL].mean()
    stats["peak_hour"] = int(hourly.idxmax())
    stats["lowest_hour"] = int(hourly.idxmin())

    # Weekday vs weekend averages
    weekday_avg = df.loc[df[DAY_TYPE_COL] == "Weekday", ENERGY_COL].mean()
    weekend_avg = df.loc[df[DAY_TYPE_COL] == "Weekend", ENERGY_COL].mean()
    stats["weekday_avg_kwh"] = weekday_avg if pd.notna(weekday_avg) else 0
    stats["weekend_avg_kwh"] = weekend_avg if pd.notna(weekend_avg) else 0

    return stats


# ======================================================================
# Hourly / peak-hour analysis
# ======================================================================

def calculate_hourly_consumption(df):
    """Average and total consumption for each hour of the day."""
    return (df.groupby(HOUR_COL)
              .agg(Avg_Power_kW=(POWER_COL, "mean"),
                   Avg_Energy_kWh=(ENERGY_COL, "mean"),
                   Total_Energy_kWh=(ENERGY_COL, "sum"),
                   Record_Count=(ENERGY_COL, "count"))
              .reset_index())


def calculate_peak_hours(df):
    """Rank hours by average consumption and classify them as Peak/Normal/Low.

    Classification method (data-driven, not manual):
      - Peak  : avg consumption >= 75th percentile of the hourly averages
      - Low   : avg consumption <= 25th percentile of the hourly averages
      - Normal: everything in between
    """
    hourly = calculate_hourly_consumption(df)
    q25 = hourly["Avg_Power_kW"].quantile(0.25)
    q75 = hourly["Avg_Power_kW"].quantile(0.75)

    def classify(avg):
        if avg >= q75:
            return "Peak"
        if avg <= q25:
            return "Low"
        return "Normal"

    hourly["Classification"] = hourly["Avg_Power_kW"].apply(classify)
    return hourly.sort_values("Avg_Power_kW", ascending=False)


def peak_hour_share(df):
    """Percentage of total energy consumed during Peak-classified hours."""
    hourly = calculate_peak_hours(df)
    peak_hours = hourly.loc[hourly["Classification"] == "Peak", HOUR_COL].tolist()
    if not peak_hours:
        return 0.0, []
    peak_mask = df[HOUR_COL].isin(peak_hours)
    share = df.loc[peak_mask, ENERGY_COL].sum() / df[ENERGY_COL].sum() * 100
    return share, peak_hours


# ======================================================================
# Weekday vs weekend analysis
# ======================================================================

def calculate_weekday_weekend(df):
    """Aggregate consumption comparison between Weekday and Weekend."""
    return (df.groupby(DAY_TYPE_COL)
              .agg(Avg_Energy_kWh=(ENERGY_COL, "mean"),
                   Total_Energy_kWh=(ENERGY_COL, "sum"),
                   Max_Energy_kWh=(ENERGY_COL, "max"),
                   Min_Energy_kWh=(ENERGY_COL, "min"),
                   Record_Count=(ENERGY_COL, "count"))
              .reset_index())


def calculate_hourly_by_day_type(df):
    """Average consumption by hour, split into Weekday / Weekend."""
    return (df.groupby([HOUR_COL, DAY_TYPE_COL])[POWER_COL]
              .mean()
              .unstack()
              .reset_index())


# ======================================================================
# Daily consumption analysis
# ======================================================================

def calculate_daily_consumption(df):
    """Daily total / average / maximum consumption."""
    daily = (df.groupby(DATE_COL)
               .agg(Total_kWh=(ENERGY_COL, "sum"),
                    Avg_kWh=(ENERGY_COL, "mean"),
                    Max_kWh=(ENERGY_COL, "max"),
                    Record_Count=(ENERGY_COL, "count"))
               .reset_index()
               .sort_values(DATE_COL))
    return daily


def detect_high_consumption_days(df, method="std"):
    """Flag unusually high-consumption days.

    Methods:
      - "std": threshold = mean + 2 x std  (days above it are unusual)
      - "iqr": upper bound = Q3 + 1.5 x IQR

    Returns (daily DataFrame with High_Consumption flag, threshold value).
    """
    daily = calculate_daily_consumption(df)
    mean = daily["Total_kWh"].mean()
    std = daily["Total_kWh"].std()
    q1 = daily["Total_kWh"].quantile(0.25)
    q3 = daily["Total_kWh"].quantile(0.75)
    iqr = q3 - q1

    if method == "iqr":
        threshold = q3 + 1.5 * iqr
        method_desc = "IQR method (Q3 + 1.5 × IQR)"
    else:
        threshold = mean + 2 * std
        method_desc = "Mean + 2 × Standard Deviation"

    daily["High_Consumption"] = daily["Total_kWh"] > threshold
    daily.attrs["threshold"] = threshold
    daily.attrs["method"] = method_desc
    return daily, threshold, method_desc


# ======================================================================
# Monthly consumption analysis
# ======================================================================

def calculate_monthly_consumption(df):
    """Total and average consumption per month (ordered Jan → Dec)."""
    monthly = (df.groupby(MONTH_COL)
                 .agg(Total_kWh=(ENERGY_COL, "sum"),
                      Avg_kWh=(ENERGY_COL, "mean"),
                      Max_kWh=(ENERGY_COL, "max"),
                      Record_Count=(ENERGY_COL, "count"))
                 .reset_index())
    monthly["Month_Num"] = monthly[MONTH_COL].map(
        {m: i + 1 for i, m in enumerate(MONTH_ORDER)})
    monthly = monthly.sort_values("Month_Num").reset_index(drop=True)
    return monthly


# ======================================================================
# Voltage vs consumption analysis
# ======================================================================

def calculate_correlations(df):
    """Correlation matrix between the main electrical quantities."""
    cols = [VOLTAGE_COL, CURRENT_COL, POWER_COL, ENERGY_COL,
            "Kitchen_Usage_kWh", "Living_Room_Usage_kWh",
            "Bedroom_Usage_kWh", "Other_Usage_kWh"]
    return df[cols].corr()


def voltage_power_correlation(df):
    """Pearson correlation between Voltage and Power."""
    return df[VOLTAGE_COL].corr(df[POWER_COL])


# ======================================================================
# Room-wise consumption analysis
# ======================================================================

def calculate_room_consumption(df):
    """Total estimated consumption per usage category."""
    rooms = {
        "Kitchen": df["Kitchen_Usage_kWh"].sum(),
        "Living Room": df["Living_Room_Usage_kWh"].sum(),
        "Bedroom": df["Bedroom_Usage_kWh"].sum(),
        "Other": df["Other_Usage_kWh"].sum(),
    }
    room_df = pd.DataFrame(list(rooms.items()), columns=["Room", "Total_kWh"])
    room_df["Share_pct"] = room_df["Total_kWh"] / room_df["Total_kWh"].sum() * 100
    return room_df


def calculate_hourly_room_consumption(df):
    """Average estimated usage per room for each hour of the day."""
    return (df.groupby(HOUR_COL)
              .agg(Kitchen=("Kitchen_Usage_kWh", "mean"),
                   Living_Room=("Living_Room_Usage_kWh", "mean"),
                   Bedroom=("Bedroom_Usage_kWh", "mean"),
                   Other=("Other_Usage_kWh", "mean"))
              .reset_index())


# ======================================================================
# High-consumption (unusual pattern) detection
# ======================================================================

def detect_high_consumption(df, method="iqr"):
    """Flag records with unusually high consumption.

    Methods:
      - "iqr": upper bound = Q3 + 1.5 × IQR of Energy_Consumption_kWh
      - "std": threshold = mean + 2 × std

    Returns (flagged DataFrame, threshold, method description).
    These records are 'Potential High-Consumption Periods' -
    unusual patterns worth investigating, NOT confirmed wastage.
    """
    df = df.copy()
    col = ENERGY_COL
    mean, std = df[col].mean(), df[col].std()
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    if method == "std":
        threshold = mean + 2 * std
        method_desc = "Mean + 2 × Standard Deviation"
    else:
        threshold = q3 + 1.5 * iqr
        method_desc = "IQR method (Q3 + 1.5 × IQR)"

    df["High_Consumption"] = df[col] > threshold
    return df, threshold, method_desc


# ======================================================================
# Day × Hour heatmap data
# ======================================================================

def calculate_day_hour_matrix(df):
    """Average consumption matrix: Day of week (rows) × Hour (columns)."""
    matrix = df.pivot_table(index=DAY_COL, columns=HOUR_COL,
                            values=POWER_COL, aggfunc="mean")
    matrix = matrix.reindex(DAY_ORDER)
    return matrix


# ======================================================================
# Key insights (computed dynamically)
# ======================================================================

def generate_insights(df):
    """Return a list of human-readable insight sentences computed from the data."""
    insights = []

    hourly = calculate_hourly_consumption(df)
    peak_hour = int(hourly.loc[hourly["Avg_Power_kW"].idxmax(), HOUR_COL])
    low_hour = int(hourly.loc[hourly["Avg_Power_kW"].idxmin(), HOUR_COL])
    insights.append(
        f"The highest average consumption occurs during **{peak_hour:02d}:00–{peak_hour + 1:02d}:00**, "
        f"and the lowest during **{low_hour:02d}:00–{low_hour + 1:02d}:00**."
    )

    wd = df.loc[df[DAY_TYPE_COL] == "Weekday", ENERGY_COL].mean()
    we = df.loc[df[DAY_TYPE_COL] == "Weekend", ENERGY_COL].mean()
    if pd.notna(wd) and pd.notna(we):
        diff_pct = (we - wd) / wd * 100
        direction = "higher" if diff_pct >= 0 else "lower"
        insights.append(
            f"Weekend consumption is **{abs(diff_pct):.1f}% {direction}** than weekday consumption "
            f"(weekend avg {we:.2f} kWh vs weekday avg {wd:.2f} kWh)."
        )

    rooms = calculate_room_consumption(df)
    top_room = rooms.loc[rooms["Total_kWh"].idxmax()]
    insights.append(
        f"The highest-consuming usage category is **{top_room['Room']}** "
        f"({top_room['Total_kWh']:.1f} kWh, {top_room['Share_pct']:.1f}% of estimated usage)."
    )

    corr = voltage_power_correlation(df)
    insights.append(
        f"The correlation between voltage and power consumption is **{corr:.3f}** "
        f"({'weak' if abs(corr) < 0.3 else 'moderate' if abs(corr) < 0.7 else 'strong'} association - "
        f"correlation does not imply causation)."
    )

    df_flagged, threshold, method = detect_high_consumption(df)
    n_high = int(df_flagged["High_Consumption"].sum())
    pct_high = n_high / len(df_flagged) * 100
    insights.append(
        f"There are **{n_high} potentially high-consumption records** ({pct_high:.1f}% of all records, "
        f"{method} method, threshold {threshold:.2f} kWh)."
    )

    monthly = calculate_monthly_consumption(df)
    if len(monthly) > 0:
        top_month = monthly.loc[monthly["Total_kWh"].idxmax()]
        low_month = monthly.loc[monthly["Total_kWh"].idxmin()]
        insights.append(
            f"The highest-consumption month is **{top_month[MONTH_COL]}** "
            f"({top_month['Total_kWh']:.1f} kWh) and the lowest is **{low_month[MONTH_COL]}** "
            f"({low_month['Total_kWh']:.1f} kWh)."
        )

    share, peak_hours = peak_hour_share(df)
    if peak_hours:
        hours_str = ", ".join(f"{h:02d}:00" for h in sorted(peak_hours))
        insights.append(
            f"**{share:.1f}%** of total energy is consumed during peak-classified hours ({hours_str})."
        )

    return insights
