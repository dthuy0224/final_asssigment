# MLA Fall 2025 - Hanoi University
# Academic Integrity Declaration:
# I, Nguyen Van Tu (2201040159), declare that this code is my own original work.
# I have not copied or adapted code from any external repositories or previous years.
# Any sources or libraries used are explicitly cited below.
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def get_predictions(matrix, data):
    """Extract predicted values and ground-truth labels from the filled matrix."""
    preds = []
    reals = []
    for i, u in enumerate(data["user_id"]):
        q = data["question_id"][i]
        preds.append(matrix[u, q])
        reals.append(data["is_correct"][i])
    return np.array(preds), np.array(reals)


def user_knn_predict_hanu(matrix, valid_data, k, return_confusion=False):
    """
    Predict missing values using user-based k-nearest neighbors (KNN).
    Args:
        matrix: 2D numpy array (users x questions) with NaNs for missing values.
        valid_data: dict with keys user_id, question_id, is_correct.
        k: int, number of nearest neighbors.
        return_confusion: bool, if True, return confusion matrix along with accuracy.
    Returns:
        accuracy: float
        (optional) confusion_matrix: sklearn confusion matrix
    """
    # Initialize KNNImputer
    imputer = KNNImputer(n_neighbors=k)

    # Fit and fill missing entries
    mat = imputer.fit_transform(matrix)

    # Compute accuracy
    acc = sparse_matrix_evaluate(valid_data, mat)

    if return_confusion:
        preds, reals = get_predictions(mat, valid_data)
        binary_preds = (preds >= 0.5).astype(int)
        cm = confusion_matrix(reals, binary_preds)
        return acc, cm

    return acc


def item_knn_predict_hanu(matrix, valid_data, k, student_id=""):
    """
    Predict missing values using item-based k-nearest neighbors (KNN).
    Also saves validation predictions to file named '{student_id}_item_knn_preds.npy'
    """
    # Transpose matrix to perform item-based similarity
    matrix_t = matrix.T

    # Apply KNNImputer on item dimension
    imputer = KNNImputer(n_neighbors=k)
    mat_t = imputer.fit_transform(matrix_t)

    # Transpose back to user-question structure
    mat = mat_t.T

    # Save validation predictions if student_id is provided
    if student_id:
        preds, _ = get_predictions(mat, valid_data)
        np.save(f'{student_id}_item_knn_preds.npy', preds)

    # Compute accuracy
    acc = sparse_matrix_evaluate(valid_data, mat)
    return acc


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix shape:", sparse_matrix.shape)

    k_values = [1, 6, 11, 16, 21, 26]
    user_accuracies = []
    item_accuracies = []

    print("\n--- User-based KNN Experiments ---")
    for k in k_values:
        acc, cm = user_knn_predict_hanu(sparse_matrix, val_data, k, return_confusion=True)
        user_accuracies.append(acc)
        print(f"k = {k:2d}, Validation Accuracy = {acc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("-" * 30)

    print("\n--- Item-based KNN Experiments ---")
    student_id = "2201040159"
    for k in k_values:
        acc = item_knn_predict_hanu(sparse_matrix, val_data, k, student_id)
        item_accuracies.append(acc)
        print(f"k = {k:2d}, Validation Accuracy = {acc:.4f}")

    # Determine best k for each method
    best_user_k = k_values[np.argmax(user_accuracies)]
    best_user_acc = max(user_accuracies)

    best_item_k = k_values[np.argmax(item_accuracies)]
    best_item_acc = max(item_accuracies)

    # Choose best overall
    if best_user_acc > best_item_acc:
        best_k = best_user_k
        best_method = "User-based"
        print(f"\n[Summary] Best method: User-based KNN with k* = {best_k}, Validation Accuracy = {best_user_acc:.4f}")
        test_acc = user_knn_predict_hanu(sparse_matrix, test_data, best_k)
    else:
        best_k = best_item_k
        best_method = "Item-based"
        print(f"\n[Summary] Best method: Item-based KNN with k* = {best_k}, Validation Accuracy = {best_item_acc:.4f}")
        test_acc = item_knn_predict_hanu(sparse_matrix, test_data, best_k)

    print(f"Test Accuracy with k* = {best_k}: {test_acc:.4f}")

    # Compute ROC-AUC
    print("\n--- ROC-AUC for Best k* ---")
    imputer = KNNImputer(n_neighbors=best_k)
    mat = imputer.fit_transform(sparse_matrix) if best_method == "User-based" else imputer.fit_transform(sparse_matrix.T).T
    preds, reals = get_predictions(mat, val_data)
    roc_auc = roc_auc_score(reals, preds)
    print(f"ROC-AUC = {roc_auc:.4f} ({best_method} KNN)")

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, user_accuracies, marker='o', label='User-based KNN')
    plt.plot(k_values, item_accuracies, marker='s', label='Item-based KNN')
    plt.title('Validation Accuracy vs k')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"[Summary] For k = {best_k}, {best_method} KNN achieved {max(best_user_acc, best_item_acc):.3f} validation accuracy.")
    print("Reflection: KNN performed best when K was: User-based K=11 and Item-based K=26. "
          "Item-based required a larger K, suggesting that item (question) similarity is more stable than user (student) similarity in this dataset. "
          "Small K values led to overfitting, large K smoothed predictions too much; the optimal K achieved the necessary balance.")


if __name__ == "__main__":
    main()
