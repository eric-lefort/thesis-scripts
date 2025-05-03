from pathlib import Path
import json
import numpy as np
from numpy.linalg import svd, lstsq, norm

import torch
import warp as wp
import plotly.graph_objects as go

from scipy.spatial.transform import Rotation as R
import open3d as o3d

from nerfstudio.utils.eval_utils import eval_setup
from rsrd.motion.motion_optimizer import RigidGroupOptimizer, RigidGroupOptimizerConfig
from rsrd.motion.atap_loss import ATAPConfig
import rsrd.transforms as tf

fig = go.Figure()

def get_optimizer(
    track_dir: Path,
) -> RigidGroupOptimizer:
    # Save the paths to the cache file.
    track_cache_path = track_dir / "cache_info.json"
    assert track_cache_path.exists()
    cache_data = json.loads(track_cache_path.read_text())
    is_obj_jointed = bool(cache_data["is_obj_jointed"])
    dig_config_path = Path(cache_data["dig_config_path"])
    track_data_path = track_dir / "keyframes.txt"

    # Load DIG model, create viewer.
    _, pipeline, _, _ = eval_setup(dig_config_path)
    try:
        pipeline.load_state()
        pipeline.reset_colors()
    except FileNotFoundError:
        print("No state found, starting from scratch")

    # Initialize tracker.
    wp.init()  # Must be called before any other warp API call.
    is_obj_jointed = False  # Unused anyway, for registration.
    optimizer_config = RigidGroupOptimizerConfig(
        atap_config=ATAPConfig(
            loss_alpha=(1.0 if is_obj_jointed else 0.1),
        ),
        altitude_down=0.0,
    )
    optimizer = RigidGroupOptimizer(
        optimizer_config,
        pipeline,
    )
    # Load keyframes.
    optimizer.load_tracks(track_data_path)
    hands = optimizer.hands_info
    assert hands is not None

    return optimizer

def compute_relative_transform(part_deltas: torch.Tensor, part_indices: tuple[int, int]):
    """
    Compute the relative pose transform between two parts at all time steps.

    Args:
        part_deltas: Tensor of shape (T, K, 7)
        part_indices: Tuple of two ints, the part indices (i, j) to compare

    Returns:
        relative_transforms: list of tf.SE3 transforms of shape (T,)
        relative_translations: (T, 3) numpy array
        relative_rotations: (T, 4) numpy array (quaternions, wxyz)
    """
    idx_a, idx_b = part_indices
    # part_a_delta = part_deltas[:, idx_a, :]  # (T, 7)
    # part_b_delta = part_deltas[:, idx_b, :]  # (T, 7)

    part_a_delta = part_deltas[idx_a, :]  # se3
    part_b_delta = part_deltas[idx_b, :]  # se3

    relative_transforms = []
    relative_translations = []
    relative_rotations = []

    print(f"part deltas shape: {part_deltas.shape}")

    for t in range(part_deltas.shape[1]):
        Ta = part_a_delta[t] # tf.SE3(part_a_delta[t])
        Tb = part_b_delta[t] # tf.SE3(part_b_delta[t])
        assert type(Ta) == tf.SE3
        assert type(Tb) == tf.SE3
        T_rel = Ta.inverse() @ Tb
        relative_transforms.append(T_rel)
        relative_translations.append(T_rel.translation().detach().cpu().numpy().squeeze())
        relative_rotations.append(T_rel.rotation().wxyz.detach().cpu().numpy().squeeze())

    return relative_transforms, np.stack(relative_translations), np.stack(relative_rotations)

def solve_hinge_joint(traj: np.ndarray):
    """
    Estimate hinge axis and pivot point from a trajectory of a 3D point.

    Args:
        traj: (T, 3) array of 3D positions of a point over time

    Returns:
        axis: (3,) unit vector (rotation axis)
        pivot: (3,) point on the hinge axis (circle center projected onto axis plane)
    """
    T = traj.shape[0]

    # Step 1: Fit a plane to the point trajectory
    center = traj.mean(axis=0)
    pts_centered = traj - center
    _, _, Vt = svd(pts_centered)
    normal = Vt[-1]  # Plane normal = least variance direction
    normal /= norm(normal)

    # Step 2: Project points onto the plane
    def project_to_plane(p, origin, normal):
        return p - np.dot(p - origin, normal) * normal

    pts_proj = np.array([project_to_plane(p, center, normal) for p in traj])

    # Step 3: Fit a circle in the plane (2D problem)
    # Choose an orthonormal basis in the plane
    x_axis = Vt[0]
    y_axis = np.cross(normal, x_axis)

    pts_2d = np.array([
        [np.dot(p - center, x_axis), np.dot(p - center, y_axis)]
        for p in pts_proj
    ])

    A = np.hstack((2 * pts_2d, np.ones((T, 1))))
    b = np.sum(pts_2d**2, axis=1)

    sol, *_ = lstsq(A, b, rcond=None)
    circle_center_2d = sol[:2]

    # Convert circle center back to 3D
    circle_center_3d = (
        center + circle_center_2d[0] * x_axis + circle_center_2d[1] * y_axis
    )

    ##########
    # --- Plotly visualization ---

    # Compute the radius of the fitted circle from the 2D points
    # (average distance from the 2D points to the fitted center)
    distances = np.linalg.norm(pts_2d - circle_center_2d, axis=1)
    radius = distances.mean()

    # Generate 100 points around the circle in 2D then convert them to 3D
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_points_3d = np.array([
        circle_center_3d + radius * np.cos(t) * x_axis + radius * np.sin(t) * y_axis
        for t in theta
    ])

    # Create a scatter for the original 3D trajectory points
    scatter_traj = go.Scatter3d(
        x=traj[:, 0],
        y=traj[:, 1],
        z=traj[:, 2],
        mode='markers',
        marker=dict(size=4, color='blue'),
        name='Trajectory Points'
    )

    # Create a scatter for the fitted circle (line)
    scatter_circle = go.Scatter3d(
        x=circle_points_3d[:, 0],
        y=circle_points_3d[:, 1],
        z=circle_points_3d[:, 2],
        mode='lines',
        line=dict(color='red', width=4),
        name='Fitted Circle'
    )

    # Create a scatter for the circle center
    scatter_center = go.Scatter3d(
        x=[circle_center_3d[0]],
        y=[circle_center_3d[1]],
        z=[circle_center_3d[2]],
        mode='markers',
        marker=dict(size=6, color='green'),
        name='Circle Center'
    )

    # Set up the figure and layout
    for e in [scatter_traj, scatter_circle, scatter_center]:
        fig.add_trace(e)
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title="Hinge Joint Circle Fit"
    )
    ##########


    return normal, circle_center_3d

def save_parts_and_hinge_html(
    means: np.ndarray,
    group_labels: np.ndarray,
    indices: tuple[int, int],
    hinge_axis: np.ndarray,
    hinge_point: np.ndarray,
    world_transforms: dict[int, list],
    save_path: Path = Path("hinge_visualization.html"),
):
    """
    Save an interactive HTML visualization of two parts, their estimated hinge joint,
    and their world-frame trajectories over time.

    Args:
        means: (N, 3) array of Gaussian centers
        group_labels: (N,) array of group IDs
        indices: tuple of part IDs to visualize
        hinge_axis: (3,) unit vector (direction)
        hinge_point: (3,) pivot point on axis
        world_transforms: dict of part_idx -> list of tf.SE3 transforms
        save_path: where to save the HTML file
    """
    p1_mask = group_labels == indices[0]
    p2_mask = group_labels == indices[1]

    means_p1 = means[p1_mask]
    means_p2 = np.hstack((means[p2_mask], np.ones((means[p2_mask].shape[0], 1))))

    # Hinge axis line
    hinge_start = hinge_point - 0.1 * hinge_axis
    hinge_end = hinge_point + 0.1 * hinge_axis
    line = np.stack([hinge_start, hinge_end])

    # Part 1 Gaussians
    fig.add_trace(go.Scatter3d(
        x=means_p1[:, 0], y=means_p1[:, 1], z=means_p1[:, 2],
        mode='markers',
        marker=dict(size=2, color='red'),
        name=f'Part {indices[0]} Gaussians'
    ))

    # Part 2 Gaussians
    fig.add_trace(go.Scatter3d(
        x=means_p2[:, 0], y=means_p2[:, 1], z=means_p2[:, 2],
        mode='markers',
        marker=dict(size=2, color='green'),
        name=f'Part {indices[1]} Gaussians'
    ))
    # Hinge axis line
    fig.add_trace(go.Scatter3d(
        x=line[:, 0], y=line[:, 1], z=line[:, 2],
        mode='lines',
        line=dict(width=6, color='blue'),
        name='Hinge Axis'
    ))

    # Pivot point
    fig.add_trace(go.Scatter3d(
        x=[hinge_point[0]], y=[hinge_point[1]], z=[hinge_point[2]],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Pivot Point'
    ))

    arrow_len = 0.05  # length of frame arrows

    # Trajectories for both parts
    for part_idx, color in zip(indices, ["red", "green"]):
        traj = np.stack([T.translation().detach().cpu().numpy().squeeze()
                        for T in world_transforms[part_idx]])
        
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
            mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(size=2),
            name=f'Part {part_idx} Trajectory'
        ))

        # Add coordinate frames every 15 frames
        for i in range(0, len(world_transforms[part_idx]), 15):
            T = world_transforms[part_idx][i]
            R = T.rotation().as_matrix().detach().cpu().numpy().squeeze()
            t = T.translation().detach().cpu().numpy().squeeze()

            # Plot X, Y, Z axes
            for axis, axis_color in zip(range(3), ["red", "green", "blue"]):
                start = t
                end = t + arrow_len * R[:, axis]
                fig.add_trace(go.Scatter3d(
                    x=[start[0], end[0]],
                    y=[start[1], end[1]],
                    z=[start[2], end[2]],
                    mode='lines',
                    line=dict(color=axis_color, width=4),
                    showlegend=False
                ))

        # Start/end markers as spheres
        fig.add_trace(go.Scatter3d(
            x=[traj[0, 0]], y=[traj[0, 1]], z=[traj[0, 2]],
            mode='markers',
            marker=dict(size=5, color='yellow'),
            name=f'Part {part_idx} Start'
        ))
        fig.add_trace(go.Scatter3d(
            x=[traj[-1, 0]], y=[traj[-1, 1]], z=[traj[-1, 2]],
            mode='markers',
            marker=dict(size=5, color='black'),
            name=f'Part {part_idx} End'
        ))

    fig.update_layout(
        title="Hinge Joint Estimate & Part Trajectories",
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=30),
        width=900,
        height=800,
    )

    fig.write_html(str(save_path))
    print(f"Visualization saved to: {save_path.resolve()}")

def rotate_about_axis(pivot, axis, angle):
    Rmat = R.from_rotvec(axis * angle).as_matrix()
    T = np.eye(4)
    T[:3, :3] = Rmat
    T[:3, 3] = pivot - Rmat @ pivot
    return T



if __name__ == "__main__":
    # Load state file
    state_file = "/home/elefort/real2sim/rsrd/outputs/cooler_5/state.pt"
    state_dict = torch.load(state_file, map_location="cpu")

    # Extract tensors
    means = state_dict["means"]
    quats = state_dict["quats"]

    for key in state_dict.keys():
        print(f"{key}: {state_dict[key].shape}")
    
    group_labels = state_dict["cluster_labels"]

    track_dir = Path("outputs/cooler_5/trajectory")
    optimizer = get_optimizer(track_dir)

    print("Elements of optimizer:")
    print(dir(optimizer))

    part_deltas = optimizer.part_deltas  # Shape: (n_frames, n_groups, 7)

    # Convert to NumPy arrays
    means_np = means.detach().numpy()
    group_labels_np = group_labels.detach().numpy()
    indices = (1, 2)  # Cluster indices for parts 1 and 2

    T_world_objinit = tf.SE3(optimizer.T_world_objinit)
    T_objreg_objinit = tf.SE3(optimizer.T_objreg_objinit.unsqueeze(0))

    n_time_steps = part_deltas.shape[0]
    n_parts = part_deltas.shape[1]
    world_transforms = [[] for _ in range(n_parts)]

    for part_idx in range(n_parts):
        T_p2o = tf.SE3(optimizer.init_p2o[part_idx].unsqueeze(0))
        for t in range(n_time_steps):
            T_pdelta = tf.SE3(optimizer.part_deltas[t, part_idx].unsqueeze(0))
            T_world = T_world_objinit @T_p2o @ T_pdelta @ T_p2o.inverse() @ T_world_objinit.inverse()
            world_transforms[part_idx].append(T_world)

    world_transforms = np.array(world_transforms)
    
    relative_transforms, relative_translations, relative_rotations = compute_relative_transform(np.array(world_transforms), indices)

    # Solve for hinge joint
    T_p2o = tf.SE3(optimizer.init_p2o[indices[0]].unsqueeze(0))
    base_frame = world_transforms[indices[0], 0] 
    #T_world_objinit @ T_objreg_objinit @ T_p2o @ tf.SE3(optimizer.part_deltas[0, indices[0]].unsqueeze(0)) @ T_p2o.inverse() @T_world_objinit.inverse()

    axis, pivot = solve_hinge_joint(relative_translations)
    pivot = base_frame @ torch.tensor(pivot)
    axis = base_frame.rotation() @ torch.tensor(axis)
    print(f"base_frame: {base_frame}")

    # to numpy
    axis =  axis.squeeze().detach().cpu().numpy()
    pivot = pivot.squeeze().detach().cpu().numpy()
    axis = axis / norm(axis)

    assert axis.shape == (3,)
    assert pivot.shape == (3,)

    # hack: adjustment
    pivot += np.array([0, 0, 0])

    # Visualize parts and hinge
    save_parts_and_hinge_html(
        means=means.detach().cpu().numpy(), #means_np,
        group_labels=group_labels_np,
        indices=indices,
        hinge_axis=axis,
        hinge_point=pivot,
        world_transforms=world_transforms,  # from convert_part_deltas_to_world
        save_path=Path("hinge_01.html")
    )

    # save hinge information to a yaml file
    hinge_info = {
        "hinge_axis": axis.tolist(),
        "hinge_point": pivot.tolist(),
        "part_indices": indices,
    }

    hinge_info_path = Path("hinge_info.json")
    hinge_info_path.write_text(json.dumps(hinge_info, indent=4))
    print(f"Hinge info saved to: {hinge_info_path.resolve()}")