"""
Analyze socket position distribution across demos in an HDF5 dataset.

This script loads socket positions from all demonstrations and creates visualizations
to understand the spatial distribution of socket placements.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
from pathlib import Path


def extract_socket_positions(hdf5_path, dataset_path="initial_state/rigid_object/socket/root_pose"):
    """
    Extract socket positions from all demos in HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
        dataset_path: Path to socket pose data within each demo
    
    Returns:
        Dictionary with demo names and their socket positions (first 3 elements of pose)
    """
    socket_positions = {}
    
    if not os.path.exists(hdf5_path):
        print(f"Error: File {hdf5_path} not found.")
        return socket_positions
    
    with h5py.File(hdf5_path, 'r') as f:
        # Check if data group exists
        if 'data' not in f:
            print("Error: 'data' group not found in HDF5.")
            return socket_positions
        
        data_group = f['data']
        demo_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')])
        
        print(f"Found {len(demo_keys)} demonstrations")
        
        for demo_key in demo_keys:
            demo = data_group[demo_key]
            
            # Only check initial_state
            socket_pose = None
            
            if dataset_path in demo:
                socket_pose = demo[dataset_path][()]
            else:
                print(f"  Warning: Could not find socket pose at {dataset_path} in {demo_key}")
                print(f"    Available keys in demo: {list(demo.keys())}")
            
            if socket_pose is not None:
                # Extract position (first 3 elements)
                # Handle both single pose and temporal sequences
                if len(socket_pose.shape) == 1:
                    pos = socket_pose[:3]
                else:
                    # Take first timestep if temporal
                    pos = socket_pose[0, :3]
                
                socket_positions[demo_key] = pos
    
    return socket_positions


def analyze_and_plot(socket_positions, output_dir="./socket_analysis"):
    """
    Create analysis plots of socket distribution.
    
    Args:
        socket_positions: Dictionary of demo names to 3D positions
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not socket_positions:
        print("No socket positions to analyze.")
        return
    
    # Convert to numpy array for analysis
    positions = np.array([pos for pos in socket_positions.values()])
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    
    print(f"\n=== Socket Position Statistics ===")
    print(f"Number of demos: {len(positions)}")
    print(f"\nX-axis (forward/backward):")
    print(f"  Min: {x.min():.4f}, Max: {x.max():.4f}, Mean: {x.mean():.4f}, Std: {x.std():.4f}")
    print(f"\nY-axis (left/right):")
    print(f"  Min: {y.min():.4f}, Max: {y.max():.4f}, Mean: {y.mean():.4f}, Std: {y.std():.4f}")
    print(f"\nZ-axis (up/down):")
    print(f"  Min: {z.min():.4f}, Max: {z.max():.4f}, Mean: {z.mean():.4f}, Std: {z.std():.4f}")
    
    print(f"\nBounding Box:")
    print(f"  X: [{x.min():.4f}, {x.max():.4f}]")
    print(f"  Y: [{y.min():.4f}, {y.max():.4f}]")
    print(f"  Z: [{z.min():.4f}, {z.max():.4f}]")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 3D Scatter Plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter = ax1.scatter(x, y, z, c=range(len(x)), cmap='viridis', s=50, alpha=0.6)
    ax1.set_xlabel('X (forward/backward)')
    ax1.set_ylabel('Y (left/right)')
    ax1.set_zlabel('Z (up/down)')
    ax1.set_title('3D Socket Position Distribution')
    plt.colorbar(scatter, ax=ax1, label='Demo Index')
    
    # 2. XY Projection (Top view)
    ax2 = fig.add_subplot(2, 3, 2)
    scatter_xy = ax2.scatter(x, y, c=range(len(x)), cmap='viridis', s=100, alpha=0.6, edgecolors='black')
    ax2.set_xlabel('X (forward/backward)')
    ax2.set_ylabel('Y (left/right)')
    ax2.set_title('XY Projection (Top View)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter_xy, ax=ax2, label='Demo Index')
    
    # Add uncertainty ellipse
    from matplotlib.patches import Ellipse
    mean_x, mean_y = x.mean(), y.mean()
    std_x, std_y = x.std(), y.std()
    ellipse = Ellipse((mean_x, mean_y), 2*std_x, 2*std_y, 
                      fill=False, edgecolor='red', linewidth=2, label='±1σ ellipse')
    ax2.add_patch(ellipse)
    ax2.plot(mean_x, mean_y, 'r+', markersize=15, markeredgewidth=2, label='Mean')
    ax2.legend()
    
    # 3. XZ Projection (Side view - forward)
    ax3 = fig.add_subplot(2, 3, 3)
    scatter_xz = ax3.scatter(x, z, c=range(len(x)), cmap='viridis', s=100, alpha=0.6, edgecolors='black')
    ax3.set_xlabel('X (forward/backward)')
    ax3.set_ylabel('Z (up/down)')
    ax3.set_title('XZ Projection (Side View)')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter_xz, ax=ax3, label='Demo Index')
    
    # 4. YZ Projection (Side view - lateral)
    ax4 = fig.add_subplot(2, 3, 4)
    scatter_yz = ax4.scatter(y, z, c=range(len(x)), cmap='viridis', s=100, alpha=0.6, edgecolors='black')
    ax4.set_xlabel('Y (left/right)')
    ax4.set_ylabel('Z (up/down)')
    ax4.set_title('YZ Projection (Lateral View)')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter_yz, ax=ax4, label='Demo Index')
    
    # 5. Histograms
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(x, bins=15, alpha=0.5, label='X', edgecolor='black')
    ax5.hist(y, bins=15, alpha=0.5, label='Y', edgecolor='black')
    ax5.hist(z, bins=15, alpha=0.5, label='Z', edgecolor='black')
    ax5.set_xlabel('Position (meters)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Position Distribution Histograms')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Statistics text box
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    stats_text = f"""
    Socket Position Statistics
    {'-'*40}
    Number of demos: {len(positions)}
    
    X-axis (forward/backward):
      Mean: {x.mean():.4f} m
      Std:  {x.std():.4f} m
      Range: [{x.min():.4f}, {x.max():.4f}]
    
    Y-axis (left/right):
      Mean: {y.mean():.4f} m
      Std:  {y.std():.4f} m
      Range: [{y.min():.4f}, {y.max():.4f}]
    
    Z-axis (up/down):
      Mean: {z.mean():.4f} m
      Std:  {z.std():.4f} m
      Range: [{z.min():.4f}, {z.max():.4f}]
    """
    ax6.text(0.1, 0.5, stats_text, fontfamily='monospace', fontsize=10,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "socket_distribution.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {plot_path}")
    plt.close()
    
    # Create 2D heatmap (density plot)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # XY heatmap
    h_xy = axes[0].hist2d(x, y, bins=20, cmap='YlOrRd')
    axes[0].set_xlabel('X (forward/backward)')
    axes[0].set_ylabel('Y (left/right)')
    axes[0].set_title('Socket Density: XY Plane (Top View)')
    plt.colorbar(h_xy[3], ax=axes[0], label='Count')
    
    # XZ heatmap
    h_xz = axes[1].hist2d(x, z, bins=20, cmap='YlOrRd')
    axes[1].set_xlabel('X (forward/backward)')
    axes[1].set_ylabel('Z (up/down)')
    axes[1].set_title('Socket Density: XZ Plane (Side View)')
    plt.colorbar(h_xz[3], ax=axes[1], label='Count')
    
    # YZ heatmap
    h_yz = axes[2].hist2d(y, z, bins=20, cmap='YlOrRd')
    axes[2].set_xlabel('Y (left/right)')
    axes[2].set_ylabel('Z (up/down)')
    axes[2].set_title('Socket Density: YZ Plane (Lateral View)')
    plt.colorbar(h_yz[3], ax=axes[2], label='Count')
    
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "socket_density_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    print(f"✅ Heatmap saved to: {heatmap_path}")
    plt.close()
    
    return positions


def save_position_csv(socket_positions, output_dir="./socket_analysis"):
    """Save socket positions to CSV for further analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "socket_positions.csv")
    
    with open(csv_path, 'w') as f:
        f.write("demo_name,x,y,z\n")
        for demo_name, pos in sorted(socket_positions.items()):
            f.write(f"{demo_name},{pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f}\n")
    
    print(f"✅ CSV saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze socket position distribution in HDF5 dataset")
    parser.add_argument('--hdf5_path', type=str, help='Path to HDF5 dataset file')
    parser.add_argument('--output_dir', type=str, default='act_copy/socket_analysis',
                       help='Directory to save analysis plots (default: act_copy/socket_analysis)')
    parser.add_argument('--save_csv', action='store_true', 
                       help='Save socket positions to CSV file')
    
    args = parser.parse_args()
    
    print(f"Loading dataset from: {args.hdf5_path}")
    socket_positions = extract_socket_positions(args.hdf5_path)
    
    if socket_positions:
        print(f"\n✅ Extracted positions from {len(socket_positions)} demos")
        analyze_and_plot(socket_positions, args.output_dir)
        
        if args.save_csv:
            save_position_csv(socket_positions, args.output_dir)
    else:
        print("❌ No socket positions found in dataset.")


if __name__ == "__main__":
    main()
