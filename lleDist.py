import pandas as pd
import numpy as np
import nolds


#Taken's Theorem allows us to reconstruct a phase space for a singular variable and calculate LLE
def calculate_lle_distribution(
    file_path: str,
    target_variable: str,
    time_variable: str,
    segment_length: int = 5000,
    overlap_percent: float = 0.5,
    embedding_dim: int = 3,
    delay_estimation_method: str = "first_zero"
):
    """
    Calculates the distribution (mean and std) of the Largest Lyapunov Exponent (LLE)
    from a time series by analyzing overlapping segments.

    Args:
        file_path (str): The path to the CSV file.
        target_variable (str): The name of the column to analyze (e.g., 'x').
        time_variable (str): The name of the time column to determine the time step.
        segment_length (int): The number of data points in each analysis segment.
                               Defaults to 5000.
        overlap_percent (float): The percentage of overlap between consecutive segments.
                                 Defaults to 0.5 (50%).
        embedding_dim (int): The embedding dimension for phase space reconstruction.
                             Defaults to 3, which is standard for the Lorenz system.
        delay_estimation_method (str): Method to estimate the time delay ('first_zero'
                                       for first zero-crossing of autocorrelation or
                                       'first_min' for first minimum of mutual information).
                                       Defaults to 'first_zero'.
    """
    # --- 1. Load and Prepare Data ---
    try:
        df = pd.read_csv(file_path)
        if target_variable not in df.columns:
            print(f"Error: Target variable '{target_variable}' not found in the CSV file.")
            return
        if time_variable not in df.columns:
            print(f"Error: Time variable '{time_variable}' not found in the CSV file.")
            return

        time_series = df[target_variable].values
        dt = np.mean(np.diff(df[time_variable].values)) # Calculate the average time step
        print(f"Successfully loaded data from '{file_path}'.")
        print(f"Analyzing target variable: '{target_variable}' with an average time step (dt) of {dt:.4f}\n")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # --- 2. Create Overlapping Segments ---
    step_size = int(segment_length * (1 - overlap_percent))
    if step_size < 1:
        step_size = 1 # Ensure we always move forward

    segments = []
    start_index = 0
    while start_index + segment_length <= len(time_series):
        segments.append(time_series[start_index : start_index + segment_length])
        start_index += step_size

    if not segments:
        print("Error: Dataset is too short for the specified segment length.")
        print(f"Time series length: {len(time_series)}, Required segment length: {segment_length}")
        return

    print(f"Created {len(segments)} overlapping segments of length {segment_length}.\n")

    # --- 3. Calculate LLE for Each Segment ---
    lle_values = []
    print("Calculating LLE for each segment...")

    # First, estimate a good time delay (lag) from the first segment
    try:
        from statsmodels.tsa.stattools import acf
        acf_vals = acf(segments[0], nlags=200, fft=True)
        lag = np.where(acf_vals <= 0)[0][0] if np.any(acf_vals <= 0) else 10
        lag = min(lag, 30)
    except Exception as e:
        print(f"Could not estimate time delay, defaulting to 10. Error: {e}")
        lag = 10

    for i, segment in enumerate(segments):
        lle = nolds.lyap_r(segment, emb_dim=embedding_dim, lag=lag, min_tsep=lag*embedding_dim)
        lle_in_nats_per_time_unit = lle * np.log(2) / dt
        lle_values.append(lle_in_nats_per_time_unit)
        print(f"  Segment {i+1}/{len(segments)} -> LLE = {lle_in_nats_per_time_unit:.4f}")

    # --- 4. Calculate and Print Final Statistics ---
    if lle_values:
        mean_lle = np.mean(lle_values)
        std_lle = np.std(lle_values) # Corrected variable name from 'le_values' to 'lle_values'

        print("\n--- LLE Calculation Summary ---")
        print(f"Mean Largest Lyapunov Exponent (LLE): {mean_lle:.4f} nats/time_unit")
        print(f"Standard Deviation of LLE: {std_lle:.4f} nats/time_unit")
        print("-------------------------------\n")
    else:
        print("Could not calculate any LLE values.")

# The call to the function is already correct
file_name = 'Chaotic_LSTM_Chebshev/Datasets/lorenz2.csv'
time_variable = 'timestep'
target_variable = 'x'
calculate_lle_distribution(file_name, target_variable, time_variable)