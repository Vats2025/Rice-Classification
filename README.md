---

# 🌾 Rice Type Classification using PyTorch

This project builds and trains a **PyTorch deep learning model** to classify rice grains based on their morphological characteristics. Using a dataset originally modified for binary classification, we distinguish between **Jasmine rice (1)** and **Gonen rice (0)** types based on extracted numerical features.

---

## 📊 About the Dataset

### 📌 Context

This dataset was created for educational purposes — to help learners practice binary classification tasks with real-world numerical data. It is a **modified version of an original rice grain dataset**.

* **Class Labels:**

  * **Jasmine → 1**
  * **Gonen → 0**


### 🧾 Content

The dataset includes rice grain samples with 11 attributes:

| Column Name       | Description                               |
| ----------------- | ----------------------------------------- |
| `id`              | Unique identifier for each sample         |
| `Area`            | Pixel area of the rice grain              |
| `MajorAxisLength` | Length of the major axis                  |
| `MinorAxisLength` | Length of the minor axis                  |
| `Eccentricity`    | Shape descriptor                          |
| `ConvexArea`      | Convex hull area                          |
| `EquivDiameter`   | Equivalent diameter                       |
| `Extent`          | Ratio of object area to bounding box area |
| `Perimeter`       | Perimeter length                          |
| `Roundness`       | Shape roundness                           |
| `AspectRation`    | Major axis / minor axis                   |
| `Class`           | 1 = Jasmine, 0 = Gonen (Target Variable)  |

---

## 🚀 Project Workflow

1. **Download Dataset**
   Used `opendatasets` and Kaggle API to fetch the dataset.

2. **Data Preprocessing**

   * Dropped `id` column
   * Removed null entries
   * Applied **max-absolute normalization** to features

3. **Split Data**

   * Train: 70%
   * Validation: 15%
   * Test: 15%

4. **PyTorch Dataset Object**
   Wrapped features and labels into a PyTorch-compatible `Dataset` object.

5. **Neural Network Architecture**
   A minimal binary classification model:

   ```plaintext
   Input Layer (10 features)
   ↓
   Hidden Layer (10 neurons, Linear + Sigmoid)
   ↓
   Output Layer (1 neuron, Sigmoid)
   ```

6. **Training and Validation**

   * Optimizer: Adam
   * Loss: Binary Cross Entropy
   * Epochs: 10
   * Achieved training accuracy \~98.7%

7. **Testing and Inference**

   * Final test accuracy: **98.79%**
   * Real-time prediction with user input (scaled using original max values)

---

## 📈 Example Performance

| Metric              | Value      |
| ------------------- | ---------- |
| Train Accuracy      | \~98.7%    |
| Validation Accuracy | \~98.2%    |
| Test Accuracy       | **98.79%** |
| Final Loss          | \~0.016    |

---

## 📂 Folder Structure

```
rice-type-classification/
├── riceClassification.csv       # Dataset file
├── rice_model.ipynb             # Full model code in notebook
├── model.pt                     # (Optional) Saved model
├── README.md                    # This file
```

---

## 🛠 Tech Stack

* Python
* PyTorch
* Pandas, NumPy
* Scikit-learn
* Matplotlib
* Google Colab
* Kaggle Dataset via OpenDatasets
