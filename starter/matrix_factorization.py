# MLA Fall 2025 - Hanoi University
# Academic Integrity Declaration:
# I, Dam Thanh Thuy (2201040173), declare that this code is my own original work.
# I have not copied or adapted code from any external repositories or previous years.
# Any sources or libraries used are explicitly cited below.

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)

def svd_reconstruct_hanu(matrix, k):

    # Copy to avoid in-place modification
    mat = matrix.astype(float).copy()

    # Impute missing values (NaN) with column means
    col_means = np.nanmean(mat, axis=0)
    inds = np.where(np.isnan(mat))
    mat[inds] = np.take(col_means, inds[1])

    # Center columns
    col_means_center = mat.mean(axis=0)
    centered = mat - col_means_center

    # SVD
    U, S, VT = np.linalg.svd(centered, full_matrices=False)
    k = min(k, S.shape[0])
    U_k = U[:, :k]
    S_k = S[:k]
    VT_k = VT[:k, :]

    # Reconstruct and de-center
    recon = (U_k * S_k) @ VT_k + col_means_center

    # Map to probability range
    recon = np.clip(recon, 0.0, 1.0)
    return recon

def squared_error_loss(data, u, z, lambda_=0.0):
    # Loss = (1/2) sum_{(n,m) in observed} (c_nm - u_n^T z_m)^2 + (lambda_/2)(||U||^2 + ||Z||^2)
    users = data["user_id"]
    questions = data["question_id"]
    correct = data["is_correct"]
    
    N = len(correct)
    total = 0.0
    
    for i in range(N):
        n = users[i]
        m = questions[i]
        c_nm = float(correct[i])
        pred = float(u[n] @ z[m])
        diff = c_nm - pred
        total += 0.5 * (diff * diff)

    data_loss = total / N
    reg = 0.5 * lambda_ * (np.sum(u * u) + np.sum(z * z))
    
    return data_loss + reg

def update_u_z(train_data, idx, lr, u, z, lambda_=0.0):

    n = train_data["user_id"][idx]
    m = train_data["question_id"][idx]
    c_nm = float(train_data["is_correct"][idx])

    u_n = u[n]
    z_m = z[m]

    pred = float(u_n @ z_m)
    diff = pred - c_nm

    # Gradients with L2 regularization
    grad_u = diff * z_m + lambda_ * u_n
    grad_z = diff * u_n + lambda_ * z_m

    # Update
    u[n] = u_n - lr * grad_u
    z[m] = z_m - lr * grad_z
    return u, z

def als(train_data, valid_data, k, lr, num_iteration, lambda_=0.01, student_id=""):

    num_users = max(train_data["user_id"]) + 1
    num_questions = max(train_data["question_id"]) + 1

    rng = np.random.default_rng(0)
    u = rng.normal(0.0, 0.1, size=(num_users, k))
    z = rng.normal(0.0, 0.1, size=(num_questions, k))

    losses = []
    val_accs = []

    N = len(train_data["is_correct"])
    
    for it in range(num_iteration):
        # FIXED: Shuffle indices at the start of each epoch
        # This ensures each observation is seen exactly once per epoch
        indices = np.random.permutation(N)
        
        # Update using shuffled indices
        for idx in indices:
            u, z = update_u_z(train_data, idx, lr, u, z, lambda_)

        # Compute metrics
        loss = squared_error_loss(train_data, u, z, lambda_)
        pred_matrix = u @ z.T
        pred_matrix = np.clip(pred_matrix, 0.0, 1.0)
        val_acc = sparse_matrix_evaluate(valid_data, pred_matrix)

        losses.append(loss)
        val_accs.append(val_acc)
        print(f"[ALS] iter={it+1}/{num_iteration} loss={loss:.4f} val_acc={val_acc:.4f}")

    fig, ax1 = plt.subplots(figsize=(7,4))
    ax1.plot(range(1, num_iteration+1), losses, label="train loss", color="tab:blue")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(range(1, num_iteration+1), val_accs, label="val acc", color="tab:orange")
    ax2.set_ylabel("val acc", color="tab:orange")
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    fig.tight_layout()
    out_name = f"mf_results_{student_id if student_id else 'hanu'}.png"
    plt.title(f"ALS (k={k}, lambda={lambda_})")
    plt.savefig(out_name, dpi=150)
    plt.close(fig)

    return (u @ z.T)

def main():
    train_matrix = load_train_sparse("./data").toarray()
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # SVD: Experiment with at least 5 k, report val accuracy, select best
    #####################################################################
    svd_ks = [10, 50, 100, 200, 500]
    svd_val_scores = {}
    svd_test_scores = {}
    for k in svd_ks:
        recon = svd_reconstruct_hanu(train_matrix, k)
        val_acc = sparse_matrix_evaluate(val_data, recon)
        test_acc = sparse_matrix_evaluate(test_data, recon)
        svd_val_scores[k] = val_acc
        svd_test_scores[k] = test_acc 
        print(f"[SVD] k={k} val_acc={val_acc:.4f} test_acc={test_acc:.4f}")

    best_k_svd = max(svd_val_scores, key=svd_val_scores.get)
    print(f"[SVD] best_k={best_k_svd} best_val_acc={svd_val_scores[best_k_svd]:.4f}")

    # ALS experiments
    als_ks = [10, 50, 100, 200, 500]
    als_val_scores = {}
    als_test_scores = {}
    student_id = "2201040173"

    base_lr = 0.01
    base_lambda = 0.01
    epochs = 15

    for k in als_ks:
        pred_matrix = als(train_data, val_data, k=k, lr=base_lr, num_iteration=epochs, lambda_=base_lambda, student_id=student_id)
        pred_matrix = np.clip(pred_matrix, 0.0, 1.0)
        val_acc = sparse_matrix_evaluate(val_data, pred_matrix)
        test_acc = sparse_matrix_evaluate(test_data, pred_matrix)
        als_val_scores[k] = val_acc
        als_test_scores[k] = test_acc
        print(f"[ALS] k={k} val_acc={val_acc:.4f} test_acc={test_acc:.4f}")

    best_k_als = max(als_val_scores, key=als_val_scores.get)
    print(f"[ALS] best_k={best_k_als} best_val_acc={als_val_scores[best_k_als]:.4f}")

    #####################################################################
    # Reflection:
    # In your report, discuss:
    # - Hyperparameter tuning process and validation
    # - Comparison of SVD and ALS
    # - Limitations of each method (esp. SVD w.r.t missing data)
    # - Effect of regularization (lambda)
    # - Plots and tables as required by assignment
    #####################################################################

    print(f"Summary: SVD best_k={best_k_svd} val={svd_val_scores[best_k_svd]:.4f}; "
          f"ALS best_k={best_k_als} val={als_val_scores[best_k_als]:.4f}. Plots saved as mf_results_{'2201040173'}.png.")

if __name__ == "__main__":
    main()
