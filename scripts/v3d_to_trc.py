# %%
import pandas as pd
import numpy as np
import os
from pathlib import Path

def read_raw_v3d_export_file(filepath:str) -> pd.DataFrame:
    """
    Reads a Visual3D export file in TSV format and converts it to a simple pandas DataFrame
    with flat column headers in the format MarkerName_Coordinate.
    
    Parameters:
    -----------
    filepath : str
        Path to the Visual3D export file (.tsv)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns named as MarkerName_Coordinate (e.g., LASIS_X)
        and frame numbers as the index.
    """
    # Read the first few lines to determine the header structure
    with open(filepath, 'r') as f:
        header_lines = [f.readline().strip() for _ in range(5)]
    
    # Extract marker names from the second line (index 1)
    filename_headers = [file for file in header_lines[0].split('\t')]
    files = list(set(filename_headers)) 

    marker_names = [name for name in header_lines[1].split('\t') if name]
    axes = [axis for axis in header_lines[4].split('\t') if axis !="ITEM"] 

    # Create column headings by combining marker names with axes
    column_headings = [f"{filename}_{marker}_{axis}" for filename, marker, axis in zip(filename_headers, marker_names, axes)]
    column_headings.insert(0,"ITEM")

    # Read the data, skipping the header rows
    df = pd.read_csv(filepath, sep='\t', skiprows=5, names=column_headings)
    
    # Rename the first column to 'Frame' and set it as index
    df.rename(columns={'ITEM': 'Frame'}, inplace=True)
    df.set_index('Frame', inplace=True)
    
    return df

def split_by_file_origin(raw_dataframe:pd.DataFrame)->pd.DataFrame:
    landmarks = raw_dataframe.reset_index()
    landmarks_long = pd.melt(landmarks,
                            id_vars=["Frame"],
                            var_name='Header',     # Name for the new column holding original column names
                            value_name='Value'       # Name for the new column holding the values
                            )
    # df_split = landmarks_long.copy()

    new_column_names = ['Origin', 'Marker_Coordinate']

    # Use .str.rsplit() to split from the right side.
    # n=2 means make at most 2 splits from the right.
    # expand=True tells pandas to return a DataFrame with separate columns.
    split_data = landmarks_long['Header'].str.rsplit('.c3d_', n=2, expand=True)
    # Assign these new columns back to the DataFrame
    landmarks_long[new_column_names] = split_data

    # Need to go back and clean up the combined Marker_Coordinate now that it's simpler to split
    new_column_names = ['Marker', 'Coordinate']
    split_data = landmarks_long['Marker_Coordinate'].str.rsplit('_',n=1, expand=True)
    landmarks_long[new_column_names] = split_data

    # Optional: Drop the original 'Header' column as it's now redundant
    landmarks_long = landmarks_long.drop(['Header', 'Marker_Coordinate'], axis=1)

    # Reorder columns for clarity
    landmarks_long = landmarks_long[['Frame', 'Origin', 'Marker', 'Coordinate', 'Value']]
    landmarks_long = landmarks_long.set_index("Frame")
    return landmarks_long

def get_all_v3d_trajectories(tsv_folder:Path, subject:str)->pd.DataFrame:
    landmarks = read_raw_v3d_export_file(Path(data_dir,f"{subject_id}_landmarks.tsv"))
    targets = read_raw_v3d_export_file(Path(data_dir,f"{subject_id}_targets.tsv"))

    print("Landmarks and Targets imported")

    # bind all the columns of data together
    trajectories = pd.concat([landmarks,targets],axis=1)

    return trajectories

def convert_df_to_trc(trajectories_df, output_filepath, frame_rate=100, units='mm'):
    """
    Convert Visual3D exported trajectories to OpenSim TRC format.
    
    Parameters:
    -----------
    trajectories_df : pandas.DataFrame
        DataFrame with columns named as MarkerName_Coordinate (e.g., LASIS_X)
    output_filepath : str or Path
        Path where the TRC file will be saved
    frame_rate : float
        Sampling frequency in Hz
    units : str
        Units for the TRC file ('mm' or 'm')
    """
    # Make a copy of the dataframe
    df = trajectories_df.copy()
    
    # Handle NaN values with interpolation
    if df.isna().any().any():
        df = df.interpolate(method='linear', limit_direction='both')
        print("NaN values interpolated")
    
    # Convert units from meters to millimeters
    df = df * 1000.0
    
    # Extract unique marker names
    markers = []
    for col in df.columns:
        # Check if the column ends with _X, _Y, or _Z
        if col.endswith('_X') or col.endswith('_Y') or col.endswith('_Z'):
            # Get everything except the last 2 characters
            marker = col[:-2]
            if marker not in markers:
                markers.append(marker)
                
                
    num_markers = len(markers)
    num_frames = len(df)
    start_frame = df.index[0]
    
    # Create time column based on frame rate
    time_column = [frame / frame_rate for frame in df.index]
    
    # Write the TRC file
    with open(output_filepath, 'w') as f:
        # Write header
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{output_filepath}\n")
        f.write(f"DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{frame_rate}\t{frame_rate}\t{num_frames}\t{num_markers}\t{units}\t{frame_rate}\t{start_frame}\t{num_frames}\n")
        
        # Write column headers - first row
        header1 = "Frame#\tTime\t"
        for marker in markers:
            header1 += f"{marker}\t\t\t"
        f.write(header1.rstrip() + "\n")
        
        # Write coordinate labels - second row
        header2 = "\t\t"
        for i in range(1, num_markers+1):
            header2 += f"X{i}\tY{i}\tZ{i}\t"
        f.write(header2.rstrip() + "\n")
        
        # Write data
        for i, frame in enumerate(df.index):
            line = f"{frame}\t{time_column[i]:.6f}\t"
            
            for marker in markers:
                # Get X, Y, Z values
                x_val = df.loc[frame, f"{marker}_X"]
                y_val = df.loc[frame, f"{marker}_Y"]
                z_val = df.loc[frame, f"{marker}_Z"]
                
                # Apply coordinate transformation from Visual3D to OpenSim
                # OpenSim: Y-up, Z-forward, X-right
                # Assuming Visual3D is: Z-up, X-forward, Y-right
                transformed_x = x_val  # Keep X as is
                transformed_y = z_val  # OpenSim Y = Visual3D Z
                transformed_z = -y_val # OpenSim Z = -Visual3D Y
                
                line += f"{transformed_x:.6f}\t{transformed_y:.6f}\t{transformed_z:.6f}\t"
            
            f.write(line.rstrip() + "\n")
    
    print(f"TRC file created: {output_filepath}")
    print(f"- Frames: {num_frames}")
    print(f"- Markers: {num_markers}")
    return None
    
# if __name__ == "__main__":
if True:
    
    subject_id = "s1"
    data_dir = r"C:\Users\Mac Prible\OneDrive - The University of Texas at Austin\research\OpenSimCourse\project\v3d_output"
    # output_dir = r"C:\Users\Mac Prible\OneDrive - The University of Texas at Austin\research\OpenSimCourse\project\s1\trc"
    output_dir = r"C:\Users\Mac Prible\repos\pdsv_opensim\output"
    output_path = Path(output_dir,f"{subject_id}_walking.trc")
    

    
    # trajectories = get_all_v3d_trajectories(Path(data_dir), subject_id)
    # convert_df_to_trc(trajectories,output_path)
    # print(trajectories.head)

# %
# %%
landmarks = read_raw_v3d_export_file(Path(data_dir,f"{subject_id}_landmarks.tsv"))
landmarks_by_origin = split_by_file_origin(landmarks)
