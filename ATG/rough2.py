import pandas as pd
import matplotlib.pyplot as plt

# Create the DataFrame
data = {
    "Generation": [899, 100],
    "Population Size": [100, 200],
    # "Mutation Rate": [0.01,0.01],
    "Best Fitness": [26.78, 0.497],
    # "Average Fitness": [0.5,0.115],
    # "Cross-over Method": ["Uniform","Order"],
}

df = pd.DataFrame(data)

# Find the row with the highest "Best Fitness"
highest_fitness_row = df["Best Fitness"].idxmax()

# Plot the table
fig, ax = plt.subplots(figsize=(10, 4))  # Adjust the figure size as needed
ax.axis("off")  # Hide the axes

# Create the table and add it to the plot
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc="center",
    cellLoc="center",
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # Scale the table size

# Add green to the header and mark the row with the highest "Best Fitness"
for (row, col), cell in table.get_celld().items():
    if row == 0:  # Header row
        cell.set_text_props(weight="bold", color="white")  # Bold and white text
        cell.set_facecolor("#4CAF50")  # Green background for header
    # elif row == highest_fitness_row + 1:  # Highlight the row with the highest "Best Fitness"
    #     cell.set_facecolor("#FFEB3B")  # Yellow background for the highlighted row

# Save the table as a PNG file
plt.savefig("compare.png", bbox_inches="tight", dpi=300)
print("Table saved as 'table.png'")
