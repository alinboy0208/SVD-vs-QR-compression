# SVD vs. QR Decomposition: Matrix Factorization for Image Compression

An interactive Python application built from scratch using **NumPy** and **Tkinter** to visualize, compare, and analyze two fundamental linear algebra techniques from **Mathematics of Data Science**: **Singular Value Decomposition (SVD)** and **QR Decomposition** in the context of image compression and dimensionality reduction.

---

## 1. Theoretical Foundation: Mathematics of Data Science

In Data Science, high-resolution images are represented as dense numerical matrices ( $A \in \mathbb{R}^{m \times n}$ per RGB color channel). Storing and processing raw pixel matrices directly is computationally expensive. 

This project applies low-rank matrix approximation and orthogonal projection from **Mathematics of Data Science** to compress visual data by retaining only the top $k$ latent features (where $k \ll \min(m, n)$ ), significantly reducing storage footprint while preserving essential structural information.

---

## 2. Applications in AI & Data Science Pipelines

Image compression via matrix factorization is not merely about saving disk space—it is a foundational preprocessing step in modern **Artificial Intelligence** and **Data Science**:

* **Dimensionality Reduction & Feature Extraction:** Compressing high-dimensional pixel data into compact latent representations speeds up training and inference for Computer Vision models while mitigating the *Curse of Dimensionality*.
* **Noise Filtering:** Minor fluctuations and high-frequency visual noise typically reside in the smallest singular values. Truncating them acts as an effective low-pass filter, yielding cleaner training data for machine learning algorithms.
* **Data Efficiency:** Enables lightweight storage and faster data transmission across distributed AI training pipelines without sacrificing critical semantic features.

---

## 3. Comparative Analysis: Why SVD Outperforms QR in Image Compression

When compressing the same image (`jpan.jpg`) down to an identical rank ( $k = 10$ ), **SVD** demonstrates vastly superior visual fidelity compared to **QR Decomposition**:

| Criterion | Singular Value Decomposition ( $A \approx U_k \Sigma_k V_k^T$ ) | QR Decomposition ( $A \approx Q_k R_k$ ) |
| :--- | :--- | :--- |
| **Compression Mechanism** | **Global Energy & Variance Prioritization:** Decomposes the matrix based on **singular values** ( $\sigma_i$ ) sorted in descending order, prioritizing axes with the sharpest color transitions, contrast, and structural variance across the entire image. | **Sequential Column & Row Space Projection:** Orthogonalizes columns sequentially from left to right (Gram-Schmidt process) without global importance sorting (no pivoting). |
| **Spatial Behavior** | Captures **global geometric structures** simultaneously across the entire frame (preserving both background silhouettes like Mt. Fuji and foreground details). | Restricts reconstruction to the subspace spanned by the **first $k$ leftmost columns**, causing horizontal streaking artifacts and completely losing central/right objects at low $k$. |
| **Optimality (Eckart–Young Theorem)** | Mathematically proven to provide the **best possible rank $k$ approximation** under both the Frobenius norm and Spectral norm. | Suboptimal for low-rank compression unless combined with column pivoting; primarily suited for solving linear least-squares problems rather than lossy data compression. |

---

## 4. Repository Structure & Quick Start

```text
SVD-vs-QR-compression/
├── svd.py       # From-scratch SVD image compressor (Eigendecomposition of A^T A)
├── qr.py        # From-scratch QR image compressor (Classical Gram-Schmidt)
├── jpan.jpg     # Sample benchmark image
└── README.md    # Project documentation & mathematical analysis

# 1. Clone the repository & navigate into the folder
git clone [https://github.com/alinboy0208/SVD-vs-QR-compression.git](https://github.com/alinboy0208/SVD-vs-QR-compression.git) && cd SVD-vs-QR-compression

# 2. Install required dependencies
pip install numpy pillow

# 3. Run the interactive visualizers (enter integer rank e.g. 10 or ratio e.g. 0.1, then press Enter)
python svd.py   # Launch SVD compressor
python qr.py    # Launch QR compressor
