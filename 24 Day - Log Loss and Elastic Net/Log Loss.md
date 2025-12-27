# Log Loss

### Log Loss in Machine Learning

**Log Loss**, also known as Logistic Loss or Cross-Entropy Loss, is a performance metric used to evaluate the accuracy of a classification model. It quantifies the difference between the predicted probabilities and the actual class labels, making it especially useful for binary and multi-class classification problems.

---

#### 1. What is Log Loss?

Log Loss measures how well the predicted probabilities of a model align with the true labels. A lower Log Loss indicates a better model.

![Screenshot 2024-07-15 at 10 51 21 PM](https://github.com/user-attachments/assets/68e0dcb5-96dc-42d0-89fd-ee712c846a0f)

![Screenshot 2024-07-15 at 10 52 36 PM](https://github.com/user-attachments/assets/1e45f0c2-a2ca-4638-8bea-2bc6c29ed4ce)

---

#### 2. Why Use Log Loss?

- **Probabilistic Interpretation**: Log Loss provides a clear measure of how confident the predictions are. A predicted probability close to 0 or 1 incurs a heavier penalty if the prediction is incorrect.
- **Sensitive to Misclassifications**: Log Loss emphasizes predictions that are wrong with high confidence, making it more sensitive to poor classifications than accuracy alone.

---

#### 3. Example Calculation

![Screenshot 2024-07-15 at 10 53 12 PM](https://github.com/user-attachments/assets/237fd42d-84e8-4cd8-a84c-ed9e0830f109)


---

#### 4. Implementing Log Loss in Python

You can calculate Log Loss using `scikit-learn`. Here's how:

```python
from sklearn.metrics import log_loss

# True labels
y_true = [0, 1, 1, 0]
# Predicted probabilities
y_pred = [0.1, 0.9, 0.8, 0.3]

# Calculate Log Loss
loss = log_loss(y_true, y_pred)
print("Log Loss:", loss)
```

---

#### 5. Interpreting Log Loss

- **Log Loss = 0**: Perfect predictions (though this is very rare).
- **Lower Log Loss**: Indicates better model performance; the model’s predicted probabilities align closely with the actual labels.
- **Higher Log Loss**: Suggests that the model's predictions are poor, especially if predictions are confidently wrong.

---

#### 6. Conclusion

Log Loss is a valuable metric for evaluating classification models, particularly when predicting probabilities. It is sensitive to the accuracy of predictions and provides a more nuanced understanding of model performance compared to simpler metrics like accuracy.
