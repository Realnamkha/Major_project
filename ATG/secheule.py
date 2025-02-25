import pandas as pd
import matplotlib.pyplot as plt

# Your data
data = [
    {
        "department": "BCT",
        "course": "F3 Drawing",
        "instructor": "Puspa Baral (09:00 - 15:00)",
        "meeting_time": "Sunday 12:30 - 2:10 (S5)",
        "section": "1"
    }
]

# Convert data to a DataFrame
df = pd.DataFrame(data)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 2))  # Adjust size as needed

# Hide axes
ax.axis('off')

# Create the table
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc='center',
    cellLoc='center'
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)

# Adjust column widths to prevent overlapping
table.auto_set_column_width([0, 1, 2, 3, 4])  # Adjust for all columns

# Add colors and formatting
for (i, j), cell in table.get_celld().items():
    if i == 0:  # Header row
        cell.set_facecolor('#4CAF50')  # Green background
        cell.set_text_props(color='white', weight='bold')  # White bold text
    else:  # Data rows
        cell.set_facecolor('#F0F0F0')  # Light gray background for data rows
        cell.set_text_props(color='black')  # Black text for data

# Add padding to cells
table.scale(1.5, 1.5)  # Increase cell size for better spacing

# Save the table as a PNG
plt.savefig('schedule_table_colorful.png', bbox_inches='tight', dpi=300)

# Show the table (optional)
plt.show()