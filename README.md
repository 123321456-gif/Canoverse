# CanoVerse
This is the repository of CanoVerse, which contains 320k canonical data from Objaverse and Objaverse-XL, along with code for orientation estimation.
<div style="text-align: center;">
  <img src="asset/teaser.png" alt="" width="100%"> 
</div>

# Overview
<div style="text-align: center;">
  <img src="asset/method.png" alt="" width="100%"> 
</div>

# Dataset
We developed a large-scale dataset containing 320k samples—a size that exceeds the total volume of all existing canonicalization datasets.
<div style="text-align: center;">
  <img src="asset/compare_to_other_dataset.png" alt="" width="100%"> 
</div>

## Comparison with [COD](https://github.com/JinLi998/CanonObjaverseDataset)
We achieve better canonicalization quality compared to COD.
<div style="text-align: center;">
  <img src="asset/cod_compare.png" alt="" width="100%"> 
</div>

We also perform inter-class alignment, whereas COD does not.

**Ours:**
<div style="text-align: center;">
  <img src="asset/inter_calss_alignment.png" alt="" width="100%"> 
</div>

**COD:**
<div style="text-align: center;">
  <img src="asset/cod_align_dataset_plot.png" alt="" width="100%"> 
</div>

Quantitative comparison between the COD dataset and our dataset (32k samples) in orientation estimation exp:

<div style="text-align: center;">
  <img src="asset/rot_estimation_exp_compare.png" alt="" width="100%"> 
</div>

## Dataset Usage

1. Please refer to the `canoverse_anno/canoverse_poses_32k.json` file to obtain the UIDs of CanoVerse.
2. Download meshes from [Objaverse](https://objaverse.allenai.org/docs/intro) using the UIDs according to their documentation.
3. Convert the file format to `.obj` files using `trimesh` or `Blender`.
4. Normalize the translation and scale. You can refer to `utils/cano_utils.py` for implementation details.
5. Apply the rotation matrix from `canoverse_anno/canoverse_poses_32k.json` to the mesh. This file also contains category information.

## Orientation Estimation

Please refer to `downstream_task/orientation_estimation/README.md`.

## Note

We only release a subset of CanoVerse. The complete dataset will be released in the future.
