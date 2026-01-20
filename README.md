


-----

# Multi-Modal 3D Object Classification using PointNet and ResNet

This repository contains a Kaggle notebook that demonstrates an end-to-end deep learning pipeline for 3D object classification. The model leverages a multi-modal approach, combining geometric data from **3D point clouds** and visual information from **2D RGB images** to achieve robust classification.

The architecture fuses features from a simplified PointNet model (for point clouds) and a pre-trained ResNet-18 model (for images), making it capable of understanding both shape and texture.

## 📦 Dataset

This project uses the **ModelNet40** dataset, which provides 3D object models from 40 different classes. The notebook specifically utilizes two modalities from the dataset:

  - **Point Clouds**: `.npy` files containing 3D coordinates for each object.
  - **RGB Images**: `.png` files representing 2D renderings of the same objects.

A key challenge addressed in the notebook is the intelligent pairing of corresponding point cloud and image files to create a cohesive multi-modal dataset.

## 🔧 Model Architecture

The neural network is composed of three primary components:

1.  **🖼️ Image Encoder (ResNet-18)**:

      - A pre-trained ResNet-18 model is used as a feature extractor.
      - The final classification layer is removed, and the model is frozen to act as a powerful image encoder.
      - It processes a `(224, 224, 3)` RGB image and outputs a **512-dimensional** feature vector representing texture and color.

2.  **🌀 Point Cloud Encoder (PointNet-style)**:

      - A simplified, PointNet-like architecture designed to process raw 3D point data.
      - It consists of 1D convolutional layers and a global max-pooling operation.
      - It takes a `(N, 3)` point cloud tensor and outputs a **512-dimensional** feature vector representing the object's geometry and spatial structure.

3.  **🔗 Fusion and Classifier Head**:

      - The feature vectors from both encoders are **concatenated** to form a single **1024-dimensional** fused feature vector.
      - This combined representation is passed through a final MLP (Multi-Layer Perceptron) classifier head to predict the object's class.

This fusion allows the model to leverage the strengths of both modalities for more accurate classification.

## 📓 Notebook Structure

The provided `3d-vision-using-point-net.ipynb` notebook is organized into a clear, step-by-step pipeline:

  - **Step 0: Setup, Imports & Configuration**

      - Imports all necessary libraries (`torch`, `numpy`, `pandas`, `glob`, etc.).
      - Defines a `Config` class to centralize key parameters like dataset paths, batch size, learning rate, and epochs.

  - **Step 1: Multi-Modal Data Discovery**

      - Scans the file system to find all point cloud files (`.npy`).
      - For each point cloud, it constructs the corresponding RGB image filename and verifies its existence.
      - All valid (point cloud, image, class) pairs are stored in a Pandas DataFrame for easy management and splitting.

  - **Step 2: The Multi-Modal Dataset Class**

      - Defines a custom PyTorch `MultiModalDataset` class.
      - This class dynamically loads a point cloud and its corresponding RGB image for a given index.
      - It also applies the necessary transformations to the image data (resizing, normalization).

  - **Step 3: The Multi-Modal Model Architecture**

      - Defines the complete neural network architecture, including the `PointCloudEncoder`, `ImageEncoder`, and the final `MultiModalNet` that fuses their outputs.

  - **Step 4: The Training & Validation Loop**

      - Implements the standard PyTorch training and validation loops.
      - The loop is adapted to handle the dictionary-based batches produced by the `MultiModalDataset`, separating point clouds and images before feeding them to the model.
      - The best-performing model based on validation accuracy is saved.

  - **Step 5: Analysis and Multi-Modal Visualization**

      - Loads the best saved model for inference.
      - Picks a random sample from the validation set and performs a prediction.
      - Visualizes both the 3D point cloud input and the 2D RGB image input side-by-side, along with the true and predicted labels for qualitative analysis.

## 🚀 How to Run

1.  **Environment**: This notebook is designed to run in a Kaggle environment with a GPU accelerator enabled.
2.  **Dataset**: Attach the [ModelNet40 dataset](https://www.google.com/search?q=https://www.kaggle.com/datasets/oyesir/modelnet40) to your Kaggle notebook. The expected file path is `/kaggle/input/modelnet40/modelnet40/`.
3.  **Execution**: Open the [notebooks/3d-vision-using-point-net.ipynb](notebooks/3d-vision-using-point-net.ipynb) notebook and run the cells sequentially from top to bottom. The code is self-contained and will handle data discovery, preprocessing, model training, and visualization.

## 📂 Project Structure

```text
3D-vision/
├── notebooks/              # Jupyter notebooks
│   └── 3d-vision-using-point-net.ipynb
├── src/                    # Source code modules
│   ├── __init__.py
│   └── utils.py
├── .gitignore              # Git ignore rules
├── README.md               # Project documentation
├── DATASET.md              # Dataset metadata
├── LICENSE                 # MIT License
├── CONTRIBUTING.md         # Contribution guidelines
└── requirements.txt        # Python dependencies
```

## 🛠️ Prerequisites

* Python 3.8+
* PyTorch
* Torchvision
* Scikit-Learn
* Plotly
* Matplotlib
* NumPy
* Pandas
* Pillow
* Tqdm


# auto-commit
