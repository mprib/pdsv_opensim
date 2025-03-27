import os
import numpy as np
import pandas as pd

def convert_to_trc(landmarks_file, targets_file, output_file, frame_rate=100):
    """
    Convert Visual3D exported marker data to OpenSim TRC format.
    
    Parameters:
    -----------
    landmarks_file : str
        Path to the landmarks file exported from Visual3D
    targets_file : str
        Path to the targets file exported from Visual3D
    output_file : str
        Path to save the output TRC file
    frame_rate : int, optional
        Frame rate of the data in Hz (default: 100)
    """
    # Read the landmark and target files
    landmarks_df = pd.read_csv(landmarks_file, sep='\t', skiprows=4)
    targets_df = pd.read_csv(targets_file, sep='\t', skiprows=4)
    
    # Extract information about the number of frames
    num_frames = len(landmarks_df)
    
    # Create time column (starting from 0 with intervals of 1/frame_rate)
    time = np.arange(num_frames) / frame_rate
    
    # Initialize the output DataFrame with Frame# and Time columns
    trc_df = pd.DataFrame()
    trc_df['Frame#'] = np.arange(1, num_frames + 1)
    trc_df['Time'] = time
    
    # Process landmark data
    landmark_markers = []
    
    # Extract column names to identify marker names
    landmark_cols = landmarks_df.columns.tolist()
    marker_cols = [col for col in landmark_cols if col not in ['ITEM']]
    
    # Parse the header rows to extract marker names
    with open(landmarks_file, 'r') as f:
        header_lines = [next(f) for _ in range(4)]
    
    # Second line contains marker names
    marker_names_line = header_lines[1].strip().split('\t')
    marker_names_line = [name for name in marker_names_line if name]
    
    # Group columns by marker (each marker has X, Y, Z)
    unique_markers = []
    for i in range(0, len(marker_names_line), 3):
        if i < len(marker_names_line):
            unique_markers.append(marker_names_line[i])
    
    # Add landmark markers to trc_df
    for marker in unique_markers:
        # Check if this marker has any data (not all NaN)
        x_col = landmark_cols[marker_cols.index(marker + '\tX') if marker + '\tX' in marker_cols else -1]
        y_col = landmark_cols[marker_cols.index(marker + '\tY') if marker + '\tY' in marker_cols else -1]
        z_col = landmark_cols[marker_cols.index(marker + '\tZ') if marker + '\tZ' in marker_cols else -1]
        
        if x_col != -1 and y_col != -1 and z_col != -1:
            # Check if there's actual data (not all NaN)
            if not pd.isna(landmarks_df[x_col]).all():
                trc_df[f'{marker}_X'] = landmarks_df[x_col]
                trc_df[f'{marker}_Y'] = landmarks_df[y_col]
                trc_df[f'{marker}_Z'] = landmarks_df[z_col]
                landmark_markers.append(marker)
    
    # Process target data
    target_markers = []
    
    # Extract column names to identify marker names
    target_cols = targets_df.columns.tolist()
    
    # Parse the header rows to extract marker names
    with open(targets_file, 'r') as f:
        header_lines = [next(f) for _ in range(4)]
    
    # Second line contains marker names
    marker_names_line = header_lines[1].strip().split('\t')
    marker_names_line = [name for name in marker_names_line if name]
    
    # Group columns by marker (each marker has X, Y, Z)
    unique_target_markers = []
    for i in range(0, len(marker_names_line), 3):
        if i < len(marker_names_line):
            unique_target_markers.append(marker_names_line[i])
    
    # Add target markers to trc_df
    for marker in unique_target_markers:
        x_idx = marker_names_line.index(marker)
        y_idx = x_idx + 1
        z_idx = x_idx + 2
        
        x_col = target_cols[x_idx]
        y_col = target_cols[y_idx]
        z_col = target_cols[z_idx]
        
        # Check if there's actual data (not all NaN)
        if not pd.isna(targets_df[x_col]).all():
            trc_df[f'{marker}_X'] = targets_df[x_col]
            trc_df[f'{marker}_Y'] = targets_df[y_col]
            trc_df[f'{marker}_Z'] = targets_df[z_col]
            target_markers.append(marker)
    
    # Get all markers (landmarks + targets)
    all_markers = landmark_markers + target_markers
    num_markers = len(all_markers)
    
    # Create and write TRC file
    with open(output_file, 'w') as f:
        # Write header
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{output_file}\n")
        f.write(f"DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{frame_rate}\t{frame_rate}\t{num_frames}\t{num_markers}\tmm\t{frame_rate}\t1\t{num_frames}\n")
        
        # Write marker names (first line)
        f.write("Frame#\tTime\t")
        for marker in all_markers:
            f.write(f"{marker}\t\t\t")
        f.write("\n")
        
        # Write marker coordinates (second line) - X1, Y1, Z1, X2, Y2, Z2...
        f.write("\t\t")
        for marker in all_markers:
            f.write(f"X{marker}\tY{marker}\tZ{marker}\t")
        f.write("\n")
        
        # Write data
        for i in range(num_frames):
            # Write frame and time
            f.write(f"{int(trc_df.loc[i, 'Frame#'])}\t{trc_df.loc[i, 'Time']:.4f}\t")
            
            # Write marker data
            for marker in all_markers:
                if f"{marker}_X" in trc_df.columns:
                    f.write(f"{trc_df.loc[i, f'{marker}_X']:.6f}\t{trc_df.loc[i, f'{marker}_Y']:.6f}\t{trc_df.loc[i, f'{marker}_Z']:.6f}\t")
                else:
                    # If marker doesn't have data for this frame, write empty values
                    f.write("\t\t\t")
            f.write("\n")
    
    print(f"TRC file successfully created: {output_file}")
    print(f"Total markers in TRC file: {num_markers}")
    print(f"Landmark markers: {landmark_markers}")
    print(f"Target markers: {target_markers}")

def process_subject(subject_id, data_dir, output_dir):
    """
    Process all files for a given subject.
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    data_dir : str
        Directory containing the Visual3D exported files
    output_dir : str
        Directory to save the output TRC files
    """
    landmarks_file = os.path.join(data_dir, f"{subject_id}_landmarks.tsv")
    targets_file = os.path.join(data_dir, f"{subject_id}_targets.tsv")
    output_file = os.path.join(output_dir, f"{subject_id}.trc")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to TRC
    convert_to_trc(landmarks_file, targets_file, output_file)

if __name__ == "__main__":
    # Example usage
    subject_id = "s1"
    data_dir = r"C:\Users\Mac Prible\OneDrive - The University of Texas at Austin\research\OpenSimCourse\project\v3d_output"
    output_dir = r"C:\Users\Mac Prible\OneDrive - The University of Texas at Austin\research\OpenSimCourse\project\trc\s1"
    
    process_subject(subject_id, data_dir, output_dir)