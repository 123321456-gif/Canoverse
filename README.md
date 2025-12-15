# CanoVerse: Canonicalization at Scale for 3D Generation and Pose

## Dataset Usage

1. Please refer to the `canoverse_anno/canoverse_poses_32k.json` file to get the UIDs of CanoVerse.
2. Download meshes from [Objaverse](https://objaverse.allenai.org/docs/intro) using the UIDs according to their documentation.
3. Convert the file format to `.obj` files using `trimesh` or `Blender`.
4. Normalize the translation and scale. You can refer to `utils/cano_utils.py` for implementation details.
5. Apply the rotation matrix from `canoverse_anno/canoverse_poses_32k.json` to the mesh. This file also contains category information.

## Orientation Estimation

Please refer to `downstream_task/orientation_estimation/README.md`.

## Note

We only release a subset of CanoVerse. We will release all data in the future.
