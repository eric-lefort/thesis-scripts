#!/usr/bin/env python3
"""
build_articulated_usd.py
------------------------
Create an articulated USD scene from two .ply meshes and hinge metadata.

Example
-------
python build_articulated_usd.py \
    --body  mesh_body.ply \
    --top   mesh_top.ply \
    --hinge hinge_info.json \
    --out   articulated.usda
"""

import argparse, json
from pathlib import Path

import numpy as np
import open3d as o3d
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Vt, Kind


# ───────────────────────────── helpers ──────────────────────────────
def load_ply(path: Path):
    """Return vertices (N×3) and faces (M×3) from a PLY file."""
    mesh = o3d.io.read_triangle_mesh(str(path))
    return np.asarray(mesh.vertices, np.float32), np.asarray(mesh.triangles, np.int32)


def add_mesh(stage, name, verts, faces, rgb=(0.5, 0.5, 0.5)):
    mesh = UsdGeom.Mesh.Define(stage, f"/World/{name}")
    points = [Gf.Vec3f(float(x), float(y), float(z))   # <-- explicit cast
              for x, y, z in verts]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
    mesh.CreateFaceVertexCountsAttr([3] * faces.shape[0])
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*rgb)]))
    UsdPhysics.RigidBodyAPI.Apply(mesh.GetPrim())          # make it a physics link
    return mesh


def add_mesh(stage, prim_path, verts, faces, color):
    """
    Build:
      /World/.../       (Xform + RigidBody)
          Mesh          (Mesh + Collision)
    Return the Xform prim.
    """
    xform = UsdGeom.Xform.Define(stage, prim_path)
    # Geometry
    mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/Mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in verts])
    mesh.CreateFaceVertexIndicesAttr([int(i) for i in faces.flatten()])
    mesh.CreateFaceVertexCountsAttr([3] * faces.shape[0])
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))

    # Physics: dynamic rigid body + triangle collider
    body_api = UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    body_api.CreateRigidBodyEnabledAttr(True)

    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())                 # mesh collider
    mass_api = UsdPhysics.MassAPI.Apply(xform.GetPrim())          # simple density
    mass_api.CreateDensityAttr(1000.0)                            # kg / m³

    return xform


def quat_from_x_to(vec):
    """Quaternion rotating +X into arbitrary unit vector `vec` (Gf.Quatf)."""
    v = np.asarray(vec, np.float32)
    v /= np.linalg.norm(v)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)

    if np.allclose(v, x_axis):
        return Gf.Quatf(1.0, 0.0, 0.0, 0.0)                       # identity
    if np.allclose(v, -x_axis):
        return Gf.Quatf(0.0, 0.0, 0.0, 1.0)                       # 180° about Z

    axis = np.cross(x_axis, v)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(x_axis, v), -1.0, 1.0))
    s = np.sin(angle * 0.5)
    return Gf.Quatf(float(np.cos(angle * 0.5)), *map(float, (axis * s)))


# ───────────────────────────── main ────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body",  type=Path, required=True, help="path to body .ply")
    ap.add_argument("--top",   type=Path, required=True, help="path to top  .ply")
    ap.add_argument("--hinge", type=Path, required=True, help="hinge_info.json")
    ap.add_argument("--out",   type=Path, default="articulated.usda")
    args = ap.parse_args()

    # Load geometry
    v_body, f_body = load_ply(args.body)
    v_top,  f_top  = load_ply(args.top)

    # Load hinge metadata
    meta  = json.loads(Path(args.hinge).read_text())
    axis  = np.asarray(meta["hinge_axis"],  np.float32)
    pivot = np.asarray(meta["hinge_point"], np.float32)
    axis /= np.linalg.norm(axis)

    # Lift entire assembly so min z ≥ 0
    all_z = np.concatenate([v_body[:, 2], v_top[:, 2]])
    dz = -float(all_z.min()) + 0.001 if all_z.min() < 0.0 else 0.0

    # Build USD stage
    stage = Usd.Stage.CreateNew(str(args.out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    if dz > 0.0:
        world.AddTranslateOp().Set(Gf.Vec3f(0, 0, dz))

    # Physics scene
    phys_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    phys_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    phys_scene.CreateGravityMagnitudeAttr().Set(9.81)

    # Rigid links
    body_xf = add_mesh(stage, "/World/MeshBody", v_body, f_body, color=(0.05, 0.15, 0.9))
    top_xf  = add_mesh(stage, "/World/MeshTop",  v_top,  f_top,  color=(0.9,  0.9,  0.9))

    UsdPhysics.ArticulationRootAPI.Apply(body_xf.GetPrim())

    # Revolute joint (1-DOF about +X in joint frame)
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/RevoluteJoint")
    joint.CreateBody0Rel().AddTarget(body_xf.GetPath())
    joint.CreateBody1Rel().AddTarget(top_xf.GetPath())
    joint.CreateAxisAttr(UsdPhysics.Tokens.x)

    # Joint frames (position + rotation)
    q = quat_from_x_to(axis)
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float, pivot)))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*map(float, pivot)))
    joint.CreateLocalRot0Attr().Set(q)
    joint.CreateLocalRot1Attr().Set(q)

    # Limits and drive
    joint.CreateLowerLimitAttr().Set(-90.0)
    joint.CreateUpperLimitAttr().Set( 90.0)

    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), UsdPhysics.Tokens.rotX)
    # drive.CreateTypeAttr(UsdPhysics.Tokens.position)      # position drive
    # drive.CreateTargetPositionAttr(0.0)
    # drive.CreateStiffnessAttr(1000.0)
    # drive.CreateDampingAttr(50.0)

    stage.GetRootLayer().Save()
    print(f"[✓] Wrote articulated USD → {args.out}")


if __name__ == "__main__":
    main()
