# 🌾 Rice Type Classification with PyTorch

This project focuses on building a **binary classifier** using PyTorch to distinguish between two types of rice: **Jasmine (1)** and **Gonen (0)** based on physical features.

---

## 📁 Dataset Overview

**Source**: [Rice Type Classification Dataset on Kaggle](https://www.kaggle.com/datasets/mssmartypants/rice-type-classification)

This dataset was created for educational and practice purposes and is based on a modified version of real-world rice grain data.

### 📌 Features in the dataset:

* `id`: Unique identifier for each sample
* `Area`: Area of the rice grain
* `MajorAxisLength`: Length of the major axis
* `MinorAxisLength`: Length of the minor axis
* `Eccentricity`: Shape descriptor of elongation
* `ConvexArea`: Area of the convex hull
* `EquivDiameter`: Equivalent diameter
* `Extent`: Ratio of object area to bounding box area
* `Perimeter`: Perimeter length
* `Roundness`: Roundness of the object
* `AspectRation`: Major axis / Minor axis ratio
* `Class`: **Target label** — `1` = Jasmine, `0` = Gonen

---

## 🛠️ Project Workflow

### 🔹 Step 4: Data Splitting

* The data is cleaned by removing `NaN` values and dropping the `id` column.
* All feature columns are normalized using **max-abs normalization** to scale values between 0 and 1, which improves model convergence.
* Labels (`Class`) are reshaped into a 1D tensor for compatibility with PyTorch loss functions.
* The dataset is split into:

  * **Training Set**: 70% (12,729 samples)
  * **Validation Set**: 15% (2,728 samples)
  * **Testing Set**: 15% (2,728 samples)
* `train_test_split` from `sklearn.model_selection` is used with a two-stage split for balanced validation and test sizes.

### 🔹 Step 5: Dataset & DataLoader

* A custom PyTorch `Dataset` class is created to handle feature-label pairing and device assignment.
* This structure simplifies batching and shuffling with `DataLoader`.
* Each data sample is automatically moved to GPU/CPU using `.to(device)` and converted to `torch.float32`.
* `DataLoader` objects are created for each data subset (train/validation/test) with a batch size of `8` to ensure efficient training.

```python
class dataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.Y = torch.tensor(Y, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
```

### 🔹 Step 6: Model Architecture

* The model is a simple **feedforward neural network** using PyTorch `nn.Module`.
* Structure:

  * **Input Layer**: 10 input features (matching the number of rice attributes)
  * **Hidden Layer**: 10 neurons (defined by `HIDDEN_NEURONS`), can be tuned for performance
  * **Output Layer**: Single neuron with `Sigmoid` activation for binary classification output
* The use of sigmoid allows output in the (0,1) range, which is suitable for binary cross-entropy loss.
* Summary shows total parameters = 121

```python
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.input_layer = nn.Linear(X.shape[1], HIDDEN_NEURONS)
        self.Linear = nn.Linear(HIDDEN_NEURONS, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.Linear(x)
        x = self.sigmoid(x)
        return x
```

### 🔹 Step 7: Training & Validation

* The training loop follows standard supervised learning steps:

  * Forward pass → Compute loss → Backward pass → Update weights
* **Loss Function**: `BCELoss` — suitable for binary classification problems
* **Optimizer**: `Adam` — known for fast convergence, learning rate = `1e-3`
* **Epochs**: `10` (can be increased for further tuning)

**Training Phase:**

* Model trained in batches using `train_dataloader`
* Predictions are rounded to 0 or 1 for accuracy computation
* `.backward()` used to compute gradients, and `optimizer.step()` applies updates

**Validation Phase:**

* Run after each epoch using `torch.no_grad()` to disable gradient calculation
* Uses `validation_dataloader` to check generalization

**Metrics Tracked Per Epoch:**

* Training loss and accuracy

* Validation loss and accuracy

* **Visualizations**:

  * Loss and accuracy curves across epochs help track overfitting or underfitting trends.

---

### 📊 Results

* **Final Test Accuracy**: `98.79%`
* Model generalizes well with very low validation and test losses.

---

## 💡 Inference

* Normalizes user inputs using max values from original (non-normalized) dataframe.
* Takes 10 inputs, passes through the model, and returns predicted class (0 or 1).

```python
model_inputs = torch.Tensor(my_inputs).to(device)
prediction = model(model_inputs)
print("Predicted Class:", round(prediction.item()))
```

---

## 📆 Credits

* Dataset credit: [Kaggle - mssmartypants](https://www.kaggle.com/datasets/mssmartypants/rice-type-classification)
---
