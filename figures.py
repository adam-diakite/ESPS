import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
dataframe = pd.read_csv('your_csv_file.csv')

# Extract the necessary columns from the DataFrame
x_values = dataframe['x_column']
y_values = dataframe['y_column']

# Create a figure and axes
fig, ax = plt.subplots()

# Plot the data
ax.plot(x_values, y_values)

# Set labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Plot from CSV data')

# Display the figure
plt.show()
