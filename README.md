# Softmax Classifier

A lightweight, educational implementation of a **multiclass Softmax classifier** (also known as multinomial logistic regression) written from scratch in pure NumPy.

This project aims to demonstrate how a linear classifier can be trained using gradient descent and the cross-entropy loss function - including support for **early stopping**, **validation monitoring**, and a scikit-learn-style interface.

---

## Features

- **Pure NumPy implementation** - no external ML frameworks required  
- **Softmax activation** for multiclass problems  
- **Cross-entropy loss** with numerical stability  
- **Early stopping** based on validation loss (`eval_set` + `patience`)  
- **Automatic class detection** (`n_classes` inferred from `y`)  
- **Compatible interface** (`fit`, `predict`, `predict_proba`, `score`)  
- **Optional verbosity** for monitoring training progress  
- **Reproducible results** with `random_state`

---

## Example Usage

```python
from my_softmax import SoftmaxClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_classification(n_samples=500, n_features=4, n_classes=3, n_informative=3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
clf = SoftmaxClassifier(learning_rate=0.1, n_epochs=1000, verbose=True)
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], patience=15)

# Evaluate performance
print("Validation accuracy:", clf.score(X_val, y_val))
