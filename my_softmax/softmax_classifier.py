import numpy as np

class SoftmaxClassifier:
  def __init__(self, learning_rate=0.1, n_epochs=1000, random_state=7, verbose=True):
    np.random.seed(random_state)
    self.learning_rate = learning_rate
    self.n_epochs = n_epochs
    self.verbose = verbose
    self.W = None
    self.best_W = None
    self.best_loss = np.inf

  @staticmethod
  def softmax(z):
    exps = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

  @staticmethod
  def one_hot(y, num_classes):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

  @staticmethod
  def cross_entropy(Y_true, Y_pred):
    epsilon = 1e-15
    Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
    return -np.sum(Y_true * np.log(Y_pred)) / Y_true.shape[0]

  def fit(self, X, y, eval_set=None, patience=10):
    """
    Training model with early stopping
    n_classes and weights W are calculated here
    """
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    if self.W is None:
      self.W = np.random.randn(n_features, n_classes)

    Y = self.one_hot(y, n_classes)
    patience_counter = 0

    if eval_set is not None:
      X_val, y_val = eval_set[0]
      Y_val = self.one_hot(y_val, n_classes)
    else:
      X_val, Y_val = None, None

    for epoch in range(self.n_epochs):
      logits = X @ self.W
      Y_pred = self.softmax(logits)
      train_loss = self.cross_entropy(Y, Y_pred)

      # gradient
      error = Y_pred - Y
      gradients = X.T @ error / n_samples
      self.W -= self.learning_rate * gradients

      if eval_set is not None:
        val_logits = X_val @ self.W
        Y_val_pred = self.softmax(val_logits)
        val_loss = self.cross_entropy(Y_val, Y_val_pred)

        if val_loss < self.best_loss:
          self.best_loss = val_loss
          self.best_W = self.W.copy()
          patience_counter = 0
        else:
          patience_counter += 1

        if self.verbose and epoch % 10 == 0:
          print(f'Epoch {epoch:4d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}')

        if patience_counter >= patience:
          if self.verbose:
            print(f'Early stopping at epoch {epoch}')
            break
      else:
        if self.verbose and epoch % 10 == 0:
          print(f'Epoch {epoch:4d} | train_loss={train_loss:4.f}')

    if self.best_W is not None:
      self.W = self.best_W.copy()

    self.n_features_ = n_features
    self.n_classes_ = n_classes
    self.is_fitted_ = True
    return self

  def predict_proba(self, X):
    if not hasattr(self, 'is_fitted_'):
      raise RuntimeError('Model must be fitted before prediction')
    logits = X @ self.W
    return self.softmax(logits)

  def predict(self, X):
    return np.argmax(self.predict_proba(X), axis=1)

  def score(self, X, y):
    return np.mean(self.predict(X) == y)