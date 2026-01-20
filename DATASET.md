# ModelNet40 Dataset

## Description
ModelNet40 is a comprehensive dataset for 3D object classification, containing 12,311 CAD models from 40 categories.

## Modalities
This project uses two modalities derived from the dataset:
1.  **Point Clouds**: `.npy` files representing 3D coordinates.
2.  **RGB Images**: `.png` files representing 2D renderings.

## License
Please refer to the [Princeton ModelNet website](http://modelnet.cs.princeton.edu/) for license details. Typically available for academic and research use.

## Source
- [Princeton ModelNet](http://modelnet.cs.princeton.edu/)
- [Kaggle Dataset](https://www.kaggle.com/datasets/oyesir/modelnet40)

## Setup
The notebook is configured to run on Kaggle. To run locally:
1.  Download the dataset from Kaggle or the official site.
2.  Ensure you have the `point_cloud/` and `rgb_imgs/` directories arranged as expected by the notebook's `Config` class.
3.  Update the `BASE_DIR` in the notebook to your local path.

# auto-commit
