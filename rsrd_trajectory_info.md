
# RSRD Object Tracking Stage Overview

This document describes how various transforms are stored, computed, and applied to Gaussian splats in the system. 

Sources:
- DIG model checkpoint (`["gauss_params"]` from `state.pth`)
- Trajectory data in JSON format (`trajectory.txt`) containing `part_deltas` and `T_objreg_objinit`.

---

## 1. Initial Gaussian Parameters

### Description
At initialization, we store the `means` and `quats` of the Gaussians from the DIG model. These represent the **original world-space positions and orientations** of each Gaussian, before any animation or motion tracking is applied.

### Code Snippet
```python
# Load DIG model state
state_file = "/home/elefort/real2sim/rsrd/outputs/cooler_5/state.pt"
state_dict = torch.load(state_file, map_location="cpu")

means = state_dict["means"]         # (N, 3)
quats = state_dict["quat"]          # (N, 4), (w, x, y, z)
labels = state_dict["cluster_labels"]

# From motion_optimizer.py
self.init_means = self.dig_model.gauss_params["means"].detach().clone()
self.init_quats = self.dig_model.gauss_params["quats"].detach().clone()
```

- **`init_means`**: Shape `(N, 3)` — original Gaussian centers.
- **`init_quats`**: Shape `(N, 4)` — original rotations in `(w, x, y, z)` format.

> Note: When passed into Warp kernels, quaternions are reordered to `(x, y, z, w)` to match `wp.quaternion(...)` input expectations.

---

## 2. Initial Object-to-World Transform — `T_world_objinit`

### Description
Defines the **initial pose** of the object in the world frame. The object is aligned with the origin and oriented with identity rotation, but translated so its centroid matches the mean of all Gaussians.

### Code Snippet
```python
self.T_world_objinit = identity_7vec()
t_centroid = self.init_means.mean(dim=0).squeeze()
self.T_world_objinit[0, 4:] = t_centroid
```

- Format: 7-vector `[w, x, y, z, tx, ty, tz]`
- Role: Establishes a consistent world frame for tracking.

---

## 3. Initial Part-to-Object Transforms — `init_p2o`

### Description
Each part (group of Gaussians) is positioned relative to the object frame. Initially, we assign identity rotations and translate each part's centroid to be relative to the object centroid.

### Code Snippet
```python
self.init_p2o = identity_7vec().repeat(self.num_groups, 1)
for i, mask in enumerate(self.group_masks):
    gp_centroid = self.init_means[mask].mean(dim=0)
    obj_centroid = self.init_means.mean(dim=0)
    offset = gp_centroid - obj_centroid
    self.init_p2o[i, 4:] = offset
```

- **`init_p2o`**: Shape `(K, 7)` — one pose per group.
- Describes the static offset of each part w.r.t. the object center at initialization.

---

## 4. Object Pose Delta — `T_objreg_objinit`

### Description
This is a single transform representing the delta between the object’s **initial pose** (e.g. as scanned) and the **registered pose** (e.g. from ZED frame registration).  
It’s applied on top of `T_world_objinit`.

### Code Snippet
```python
self.T_objreg_objinit = torch.tensor(data["T_objreg_objinit"]).cuda()
```

- Format: `(7,)` vector `[w, x, y, z, tx, ty, tz]`
- Applies as a **delta in world coordinates**.

> 🔁 Interpreted as: `T_world_objreg = T_world_objinit * T_objreg_objinit`

---

## 5. Part Pose Deltas — `part_deltas`

### Description
These are **time-varying pose updates** per part group. They describe how each part moves over time, relative to its initial pose.

### Code Snippet
```python
self.part_deltas = torch.nn.Parameter(
    torch.tensor(data["part_deltas"]).cuda()
)
```

- Shape: `(T, K, 7)` where:
  - `T`: number of timesteps
  - `K`: number of parts
- These are directly optimized during tracking to fit observations.

---

## 6. Applying Transforms in Warp Kernel

### Description
The Warp kernel `apply_to_model_warp` computes the final **world-space pose of each Gaussian** using the composed transformation chain.

### Transform Chain per Gaussian

For Gaussian `i` in group `g`:
```python
new_g2w_T = (
    o2w_T        # object-to-world (initial)
    * odelta_T   # object delta (registered pose)
    * p2o_T      # part-to-object (initial)
    * pdelta_T   # per-frame part delta
    * g2p_T      # gaussian-to-part (local offset)
)
```

Where:
- `g2w_T` is the original transform from `init_means`, `init_quats`
- `g2p_T = inverse(p2o_T) * inverse(o2w_T) * g2w_T`  
  → This computes each Gaussian’s local frame in part coordinates.

### Kernel Snippet
```python
@wp.kernel
def apply_to_model_warp(...):
    ...
    o2w_T = poses_7vec_to_transform(init_o2w, 0)
    p2o_T = poses_7vec_to_transform(init_p2os, group_id)
    odelta_T = poses_7vec_to_transform(o_delta, 0)
    pdelta_T = poses_7vec_to_transform(p_deltas, group_id)
    
    g2w_T = wp.transformation(means[tid], quat_from_array(quats[tid]))
    g2p_T = wp.transform_inverse(p2o_T) * wp.transform_inverse(o2w_T) * g2w_T

    new_g2w_T = o2w_T * odelta_T * p2o_T * pdelta_T * g2p_T

    means_out[tid] = wp.transform_get_translation(new_g2w_T)
    quats_out[tid] = wp.transform_get_rotation(new_g2w_T)
```

---

## Summary of Transform Storage and Flow

| Component              | Symbol                | Shape         | Description                                 |
|------------------------|------------------------|---------------|---------------------------------------------|
| Gaussian means         | `init_means`           | `(N, 3)`      | World-space positions of splats             |
| Gaussian rotations     | `init_quats`           | `(N, 4)`      | World-space rotations (wxyz)                |
| Object initial pose    | `T_world_objinit`      | `(1, 7)`      | Aligns object centroid to origin            |
| Part initial poses     | `init_p2o`             | `(K, 7)`      | Each part's offset from object              |
| Object pose delta      | `T_objreg_objinit`     | `(7,)`        | Tracks object movement                      |
| Part motion deltas     | `part_deltas`          | `(T, K, 7)`   | Tracks part movement over time              |

---

## Notes
- Transform composition is handled with `wp.Transformation` in Warp.
- Pose 7-vectors are consistently used in `[w, x, y, z, tx, ty, tz]` format.
- Warp kernels manage reordering where necessary (`wp.quaternion(x, y, z, w)`).

---

This overview clarifies how 3D Gaussian transforms are structured and composed to animate an object and its parts in space across time.


