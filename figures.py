import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import roc_curve, auc

preds = pd.read_csv('/home/adamdiakite/Documents/ESPS-main/HDF5_Test/predictions.csv')
preds_no_tresh = pd.read_csv('/home/adamdiakite/Documents/ESPS-main/HDF5_Test/predictions_no_tresh.csv')
preds_strides = pd.read_csv('/home/adamdiakite/Documents/ESPS-main/HDF5_Test/predictions_strides.csv')
preds_uniform = pd.read_csv('/home/adamdiakite/Documents/ESPS-main/HDF5_Test/predictions_uniform.csv')


def plot_score_distribution():
    """
    PLots the score distribution versus the surface for all patients, on after another.
    :return:
    """
    # Load the CSV file

    # Select the desired columns for plotting
    slice_score_columns = [col for col in preds.columns if col.startswith('Slice') and not col.endswith('_surface')]
    slice_surface_columns = [col for col in preds.columns if col.startswith('Slice') and col.endswith('_surface')]

    # Iterate over the rows (patients) in the DataFrame
    for _, row in preds.iterrows():
        patient_id = row['ID']
        slice_scores = row[slice_score_columns].values
        slice_surfaces = row[slice_surface_columns].values

        # Create subplots with 1 row and 2 columns
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Scatter plot: slice scores vs slice surfaces
        axs[0].scatter(slice_surfaces, slice_scores)
        axs[0].set_xlabel('Slice Surface')
        axs[0].set_ylabel('Slice Score')
        axs[0].set_title(f'Slice Scores vs Slice Surfaces for Patient {patient_id}')

        # Histogram: slice scores
        axs[1].hist(slice_scores, bins=10, edgecolor='black')
        axs[1].set_xlabel('Slice Score')
        axs[1].set_ylabel('Frequency')
        axs[1].set_title(f'Slice Score Distribution for Patient {patient_id}')

        # Adjust spacing between subplots
        plt.tight_layout()

        # Show the plots
        plt.show()


def plot_score_distribution(preds, patient_id):
    """Plots the score distribution for a single patient.
    :param preds: HDF5 file containing all the patients
    :param patient_id: Specific patient i want to plot
    :return:
    """
    # Select the desired columns for plotting
    slice_score_columns = [col for col in preds.columns if col.startswith('Slice') and not col.endswith('_surface')]
    slice_surface_columns = [col for col in preds.columns if col.startswith('Slice') and col.endswith('_surface')]

    # Find the row for the specified patient ID
    row = preds[preds['ID'] == patient_id]

    # Check if the patient ID exists in the DataFrame
    if row.empty:
        print(f"Patient {patient_id} not found.")
        return

    # Extract the slice scores and surfaces for the patient
    slice_scores = row[slice_score_columns].values.flatten()
    slice_surfaces = row[slice_surface_columns].values.flatten()

    # Create subplots with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Scatter plot: slice scores vs slice surfaces
    axs[0].scatter(slice_surfaces, slice_scores)
    axs[0].set_xlabel('Slice Surface')
    axs[0].set_ylabel('Slice Score')
    axs[0].set_title(f'Slice Scores vs Slice Surfaces for Patient {patient_id}')

    # Histogram: slice scores
    axs[1].hist(slice_scores, bins=10, edgecolor='black')
    axs[1].set_xlabel('Slice Score')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title(f'Slice Score Distribution for Patient {patient_id}')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()


def plot_correlation_matrix():
    """
    Plots correlation matrix for all the data in the CSV file. (minus the slice score and surface.)
    :return:
    """
    # Load the CSV file
    preds = pd.read_csv('/home/adamdiakite/Documents/ESPS-main/HDF5_Test/predictions.csv')

    preds_filtered = preds.loc[:, ~preds.columns.str.contains('Slice')]

    # Calculate the correlation matrix
    correlation_matrix = preds_filtered.corr()

    # Convert the correlation matrix to a NumPy array
    corr_array = correlation_matrix.values

    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)

    print(correlation_matrix)
    # Plot the correlation matrix as a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_array, cmap='RdYlBu')
    plt.colorbar()
    plt.xticks(np.arange(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(np.arange(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title('Correlation Matrix')
    plt.show()


def plot_scores_surfaces_all(preds):
    """
    Slice score vs Slice surface, automatically on all patients one after another
    :param preds:
    :return:
    """
    # Select the columns for slice scores and surfaces
    slice_score_columns = [col for col in preds.columns if col.startswith('Slice') and not col.endswith('_surface')]
    slice_surface_columns = [col for col in preds.columns if col.startswith('Slice') and col.endswith('_surface')]

    # Iterate over each row in the dataset
    for _, patient_data in preds.iterrows():
        # Get the patient ID
        patient_id = patient_data['ID']

        # Get the slice numbers
        slice_numbers = [col.split(' ')[1] for col in slice_score_columns]

        # Get the slice scores and surfaces for the patient
        slice_scores = [patient_data[col] for col in slice_score_columns]
        slice_surfaces = [patient_data[col] for col in slice_surface_columns]

        # Calculate the total tumor volume
        total_volume = np.nansum(slice_surfaces)

        # Create the figure and axes
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot the slice scores
        ax1.plot(slice_numbers, slice_scores, 'bo-', label='Slice Score')
        ax1.set_xlabel('Slice Number')
        ax1.set_ylabel('Slice Score')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Create a twin axes for slice surfaces
        ax2 = ax1.twinx()

        # Plot the slice surfaces
        ax2.plot(slice_numbers, slice_surfaces, 'ro-', label='Slice Surface')
        ax2.set_ylabel('Slice Surface')
        ax2.tick_params(axis='y', labelcolor='red')

        # Add legend for total tumor volume
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

        # Set title with total tumor volume in green
        plt.title(f'Slice Scores and Surfaces for Patient {patient_id}\nTotal Tumor Volume: {total_volume}',
                  color='green')

        # Rotate x-axis labels if needed
        plt.xticks(rotation=90)

        # Show the plot
        plt.show()


def plot_scores_surfaces(preds, patient_id):
    """
    Slice score vs slice surface for a specified patient in parameters.
    :param preds:
    :param patient_id:
    :return:
    """
    # Select the columns for slice scores and surfaces
    slice_score_columns = [col for col in preds.columns if col.startswith('Slice') and not col.endswith('_surface')]
    slice_surface_columns = [col for col in preds.columns if col.startswith('Slice') and col.endswith('_surface')]

    # Find the row for the specified patient ID
    patient_data = preds[preds['ID'] == patient_id].iloc[0]

    # Get the slice numbers
    slice_numbers = [col.split(' ')[1] for col in slice_score_columns]

    # Get the slice scores and surfaces for the patient
    slice_scores = [patient_data[col] for col in slice_score_columns]
    slice_surfaces = [patient_data[col] for col in slice_surface_columns]

    # Calculate the total tumor volume
    total_volume = np.nansum(slice_surfaces)

    # Create the figure and axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the slice scores
    ax1.plot(slice_numbers, slice_scores, 'bo-', label='Slice Score')
    ax1.set_xlabel('Slice Number')
    ax1.set_ylabel('Slice Score')
    ax1.set_ylim(0, 1)  # Set the y-axis limits to 0 and 1 for slice scores
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a twin axes for slice surfaces
    ax2 = ax1.twinx()

    # Plot the slice surfaces
    ax2.plot(slice_numbers, slice_surfaces, 'ro-', label='Slice Surface')
    ax2.set_ylabel('Slice Surface')
    ax2.tick_params(axis='y', labelcolor='red')

    # Add legend for total tumor volume
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

    # Set title with total tumor volume in green
    plt.title(f'Slice Scores and Surfaces for Patient {patient_id}\nTotal Tumor Volume: {total_volume}', color='green')

    # Rotate x-axis labels if needed
    plt.xticks(rotation=90)

    # Show the plot
    plt.show()



def various_plots(data):
    """
    Scatter plot of 'max surface' versus 'duree_tki' for all patients in the CSV.
    :param data: DataFrame containing the data
    """
    # Extract the 'max surface' and 'duree_tki' columns
    max_surface = data['max score']
    duree_tki = data['duree_tki']

    # Create the scatter plot
    plt.scatter(duree_tki, max_surface)
    plt.xlabel('Tumor volume')
    plt.ylabel('max prediction score')
    plt.title('max prediction score on all slices vs tumor volume for all patients')

    # Show the plot
    plt.show()


def scatter_plot_groups(preds):
    """
    Scatter plot of 'min score' versus 'duree_tki' for two groups of patients based on 'duree_tki' value.
    :param preds: DataFrame containing the data
    """
    # Create two groups based on 'duree_tki' value
    above_273 = preds[preds['duree_tki'] > 273]
    below_273 = preds[preds['duree_tki'] <= 273]

    # Plot the scatter plot for both groups
    plt.scatter(above_273['duree_tki'], above_273['min score'], color='blue', label='Good responders')
    plt.scatter(below_273['duree_tki'], below_273['min score'], color='red', label='Bad responders')

    # Set the labels and title
    plt.xlabel('Durée TKI')
    plt.ylabel('Minimum Score')
    plt.title('Scatter Plot: Minimum score vs Durée TKI')

    # Add the legend
    plt.legend()

    # Show the plot
    plt.show()




# plot_correlation_matrix()


# Un seul affichage pour les trois
plot_scores_surfaces(preds_uniform, '2-21-0049')
#plot_scores_surfaces(preds_uniform, '2-21-0049')
# plot_scores_surfaces(preds_strides, '2-21-0060')

# Un seul affichage pour les trois

#various_plots(preds)
#various_plots(preds_uniform)
# various_plots(preds_strides)

# Un seul affichage pour les trois
#scatter_plot_groups(preds_uniform)
# scatter_plot_groups(preds_no_tresh)
# scatter_plot_groups(preds_strides)


# plot_correlation_matrix()
