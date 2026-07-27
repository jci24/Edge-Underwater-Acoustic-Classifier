import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

CATALOGUE_FILE = "data/catalogues/deepship_recordings.csv"
OUTPUT_FILE = "data/plots/deepship_class_distribution.png"


# Read the metadata table.
recordings = pd.read_csv(CATALOGUE_FILE)

# Count files in each class.
file_counts = recordings.groupby("class").size()

# Calculate total duration in each class.
duration_seconds = recordings.groupby("class")["duration_seconds"].sum()
duration_minutes = duration_seconds / 60

# Count unique vessels in each class.
source_group_counts = recordings.groupby("class")["vessel_name"].nunique()

# Put the results into one table.
class_summary = pd.DataFrame(
    {
        "Files": file_counts,
        "Duration (minutes)": duration_minutes,
        "Source groups": source_group_counts,
    }
)

print(class_summary)

# Create three side-by-side plots.
figure, axes = plt.subplots(1, 3, figsize=(15, 5))

class_summary["Files"].plot(
    kind="bar",
    ax=axes[0],
    color="steelblue",
    title="Files by class",
)

class_summary["Duration (minutes)"].plot(
    kind="bar",
    ax=axes[1],
    color="darkorange",
    title="Duration by class",
)

class_summary["Source groups"].plot(
    kind="bar",
    ax=axes[2],
    color="seagreen",
    title="Unique vessels by class",
)

axes[0].set_ylabel("Number of files")
axes[1].set_ylabel("Duration in minutes")
axes[2].set_ylabel("Number of unique vessels")

for axis in axes:
    axis.set_xlabel("Ship class")
    axis.tick_params(axis="x", rotation=30)

figure.suptitle("DeepShip public dataset class distribution")
figure.tight_layout()

# Create the output folder and save the plot.
Path("data/plots").mkdir(parents=True, exist_ok=True)
figure.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")

plt.show()

print(f"Saved plot to {OUTPUT_FILE}")
