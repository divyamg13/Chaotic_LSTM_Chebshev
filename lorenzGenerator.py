import numpy as np
from scipy.integrate import odeint
import csv
import sys
import os

def lorenz(state, t, sigma, rho, beta):
    """
    Defines the system of Lorenz differential equations.
    """
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

def generate_lorenz_data(n, sigma=10, rho=28, beta=8/3, x0=0.0, y0=1.0, z0=0.0):
    """
    Generates n samples of the Lorenz attractor.
    
    Args:
        n (int): The number of samples to generate.
        sigma (float): The Prandtl number.
        rho (float): The Rayleigh number.
        beta (float): A constant parameter.
        x0 (float): Initial condition for x.
        y0 (float): Initial condition for y.
        z0 (float): Initial condition for z.

    Returns:
        A numpy array with columns for time (timestep), x, y, and z.
    """
    # Create an integer timestep array from 1 to n
    t = np.arange(0, n * 0.01, 0.01)
    
    # We use a separate floating-point time vector for the odeint solver.

    initial_state = [x0, y0, z0]
    
    # Use odeint to numerically solve the differential equations.
    solution = odeint(lorenz, initial_state, t, args=(sigma, rho, beta))
    
    # Add the integer timestep as the first column to the solution array.
    timestep_column = t.reshape(-1, 1)
    
    # Use np.hstack to horizontally stack the timestep and the solution.
    data_with_timestep = np.hstack((timestep_column, solution))

    return data_with_timestep

def save_to_csv(data, folder, filename):
    """
    Saves the generated data to a CSV file in a specified folder.
    
    Args:
        data (numpy.ndarray): The data to save.
        folder (str): The name of the folder to save to.
        filename (str): The name of the CSV file.
    """
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created directory: {folder}")
        
    # Construct the full file path
    filepath = os.path.join(folder, filename)
    
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        # The header row now includes 'timestep'
        writer.writerow(['timestep', 'x', 'y', 'z'])
        writer.writerows(data)
    print(f"Data saved successfully to {filepath}")

# --- Main Execution ---
if __name__ == '__main__':
    # Check if a filename was provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python lorenz_script.py <output_filename.csv>")
        sys.exit(1)
    
    # Get the filename from the command line
    output_filename = sys.argv[1]
    
    # The folder where the data will be saved
    output_folder = "Datasets"
    
    # User-defined parameters and initial conditions
    num_samples = 20000
    sigma_val = 10
    rho_val = 28
    beta_val = 8/3
    x0_val = -1.1
    y0_val = 0.1
    z0_val = 0.9

    # 1. Generate the Lorenz data, now with an integer timestep
    lorenz_data = generate_lorenz_data(
        num_samples,
        sigma=sigma_val,
        rho=rho_val,
        beta=beta_val,
        x0=x0_val,
        y0=y0_val,
        z0=z0_val
    )

    # 2. Save the data to the specified CSV file in the designated folder
    save_to_csv(lorenz_data, output_folder, output_filename)
    
    print("Script finished. No plot was generated.")