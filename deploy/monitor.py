import json
import os

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Load reference and current data
ref_data = pd.read_csv("../data/monitoring/monitor_ref.csv")
curr_data = pd.read_csv("../data/monitoring/monitor_curr.csv")

# Rename the columns to 'target' and 'prediction'
ref_data = ref_data.rename(columns={"median_house_value": "target"})
curr_data = curr_data.rename(columns={"median_house_value": "target"})

# Create report
report = Report(
    metrics=[
        DataDriftPreset(),
        # RegressionPreset()
    ]
)

# Run report
res = report.run(reference_data=ref_data, current_data=curr_data)

# Save results as JSON and HTML
os.makedirs("../data/monitoring", exist_ok=True)
res.save_json("../data/monitoring/monitoring_report.json")

print("Reports saved")

# Check for performance drift based on drift share threshold (e.g., 0.5 = 50%)
with open("../data/monitoring/monitoring_report.json", "r") as f:
    report_data = json.load(f)

# Default to no drift
drift_detected = False

# Look for the DriftedColumnsCount metric
for metric in report_data["metrics"]:
    if metric["metric_id"].startswith("DriftedColumnsCount"):
        drift_share = metric["value"]["share"]
        drift_threshold = 0.5  # Customize this threshold as needed
        drift_detected = drift_share > drift_threshold
        break

# Save drift status to a file
with open("../data/monitoring/drift_status.txt", "w") as drift_file:
    drift_file.write("Drift Detected" if drift_detected else "No Drift")

print(
    "Drift check complete. Status:",
    "Drift Detected" if drift_detected else "No Drift",
)
