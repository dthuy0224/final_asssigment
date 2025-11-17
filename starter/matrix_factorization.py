# MLA Fall 2025 - Hanoi University
# Academic Integrity Declaration:
# I, Dam Thanh Thuy (2201040173), declare that this code is my own original work.
# I have not copied or adapted code from any external repositories or previous years.
# Any sources or libraries used are explicitly cited below.

import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)

def svd_reconstruct_hanu(matrix, k, zero_mask=False):

    # Copy to avoid in-place modification
    mat = matrix.astype(float).copy()

    # Impute missing values (NaN) with column means or zeros
    if zero_mask:
        # Zero imputation
        mat = np.nan_to_num(mat, nan=0.0)
        col_means = np.zeros(mat.shape[1])
    else:
        # Mean imputation (default)
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

def als(train_data, valid_data, k, lr, num_iteration, lambda_=0.01, student_id="", plot=True):

    num_users = max(train_data["user_id"]) + 1
    num_questions = max(train_data["question_id"]) + 1

    rng = np.random.default_rng(0)
    u = rng.normal(0.0, 0.1, size=(num_users, k))
    z = rng.normal(0.0, 0.1, size=(num_questions, k))

    losses = []
    val_accs = []

    N = len(train_data["is_correct"])
    
    for it in range(num_iteration):
        indices = np.random.permutation(N)
        for idx in indices:
            u, z = update_u_z(train_data, idx, lr, u, z, lambda_)

        # Compute metrics
        loss = squared_error_loss(train_data, u, z, lambda_)
        pred_matrix = u @ z.T
        pred_matrix = np.clip(pred_matrix, 0.0, 1.0)
        val_acc = sparse_matrix_evaluate(valid_data, pred_matrix)

        losses.append(loss)
        val_accs.append(val_acc)
        if plot:
            print(f"[ALS] iter={it+1}/{num_iteration} loss={loss:.4f} val_acc={val_acc:.4f}")

    if plot:
        fig, ax1 = plt.subplots(figsize=(7,4))
        ax1.plot(range(1, num_iteration+1), losses, label="train loss", color="tab:blue")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss", color="tab:blue")
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax2 = ax1.twinx()
        ax2.plot(range(1, num_iteration+1), val_accs, label="val acc", color="tab:orange")
        ax2.set_ylabel("val acc", color="tab:orange")
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        fig.suptitle(f"ALS (k={k}, lambda={lambda_})", fontsize=12, fontweight='bold', y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_name = f"mf_results_{student_id if student_id else 'hanu'}.png"
        plt.savefig(out_name, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return (u @ z.T), losses, val_accs

def main():
    train_matrix = load_train_sparse("./data").toarray()
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    # Part (a): SVD experiments
    print("="*80)
    print("SVD EXPERIMENTS")
    print("="*80)
    svd_ks = [10, 50, 100, 200, 500]
    svd_val_scores = {}
    svd_test_scores = {}
    for k in svd_ks:
        recon = svd_reconstruct_hanu(train_matrix, k)
        val_acc = sparse_matrix_evaluate(val_data, recon)
        test_acc = sparse_matrix_evaluate(test_data, recon)
        svd_val_scores[k] = val_acc
        svd_test_scores[k] = test_acc 
        print(f"k={k:3d}: val={val_acc:.4f}, test={test_acc:.4f}")

    best_k_svd = max(svd_val_scores, key=svd_val_scores.get)
    print(f"\nBest SVD: k={best_k_svd}, val={svd_val_scores[best_k_svd]:.4f}, test={svd_test_scores[best_k_svd]:.4f}")

    # Part (c): ALS Hyperparameter Tuning
    print("\n" + "="*80)
    print("ALS HYPERPARAMETER TUNING")
    print("="*80)
    student_id = "2201040173"
    als_ks = [10, 50, 100, 200, 500]
    learning_rates = [0.001, 0.01, 0.05]
    lambdas = [0.001, 0.01, 0.1]
    num_epochs_list = [10, 15, 20]
    
    best_configs = {}
    for k in als_ks:
        print(f"\n[TUNING] k={k}")
        print("-" * 80)
        best_val_acc = -1
        best_config = None
        
        for lr in learning_rates:
            for lambda_ in lambdas:
                for num_epochs in num_epochs_list:
                    pred_matrix, _, _ = als(
                        train_data, val_data, k=k, lr=lr, 
                        num_iteration=num_epochs, lambda_=lambda_, 
                        student_id=student_id, plot=False
                    )
                    pred_matrix = np.clip(pred_matrix, 0.0, 1.0)
                    val_acc = sparse_matrix_evaluate(val_data, pred_matrix)
                    test_acc = sparse_matrix_evaluate(test_data, pred_matrix)
                    
                    print(f"  lr={lr:.3f}, lambda={lambda_:.3f}, epochs={num_epochs:2d} | "
                          f"val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_config = (lr, lambda_, num_epochs, val_acc, test_acc)
        
        best_configs[k] = best_config
        print(f"\n[Best for k={k}] lr={best_config[0]:.3f}, "
              f"lambda={best_config[1]:.3f}, epochs={best_config[2]:2d} | "
              f"val_acc={best_config[3]:.4f}, test_acc={best_config[4]:.4f}")
    
    best_k_overall = max(best_configs.keys(), key=lambda k: best_configs[k][3])
    best_overall_config = best_configs[best_k_overall]
    
    print("\n" + "="*80)
    print("BEST OVERALL CONFIGURATION")
    print("="*80)
    print(f"k={best_k_overall}, lr={best_overall_config[0]:.3f}, "
          f"lambda={best_overall_config[1]:.3f}, epochs={best_overall_config[2]:2d}")
    print(f"Validation Accuracy: {best_overall_config[3]:.4f}")
    print(f"Test Accuracy: {best_overall_config[4]:.4f}")
    
    print("\n[Training best model with plotting...]")
    pred_matrix, losses, val_accs = als(
        train_data, val_data,
        k=best_k_overall, 
        lr=best_overall_config[0], 
        num_iteration=best_overall_config[2],
        lambda_=best_overall_config[1], 
        student_id=student_id, 
        plot=True
    )
    
    # Persist best configuration for reuse by other modules (e.g., ensemble.py)
    best_config_payload = {
        "student_id": student_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "best_config": {
            "k": best_k_overall,
            "lr": best_overall_config[0],
            "lambda": best_overall_config[1],
            "epochs": best_overall_config[2],
            "val_acc": best_overall_config[3],
            "test_acc": best_overall_config[4],
        },
        "best_by_k": {
            str(k): {
                "lr": cfg[0],
                "lambda": cfg[1],
                "epochs": cfg[2],
                "val_acc": cfg[3],
                "test_acc": cfg[4],
            }
            for k, cfg in best_configs.items()
        },
    }
    with open("mf_best_config.json", "w", encoding="utf-8") as f:
        json.dump(best_config_payload, f, indent=2)
    print("\n[Saved best ALS config -> mf_best_config.json]")

    # Summary table
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("="*80)
    print(f"{'k':<6} {'lr':<8} {'lambda':<8} {'epochs':<8} {'val_acc':<10} {'test_acc':<10}")
    print("-" * 80)
    for k in als_ks:
        config = best_configs[k]
        print(f"{k:<6} {config[0]:<8.3f} {config[1]:<8.3f} {config[2]:<8} "
              f"{config[3]:<10.4f} {config[4]:<10.4f}")
    
    # Part (d): Comparison with SVD
    print("\n" + "="*80)
    print("COMPARISON: SVD vs ALS")
    print("="*80)
    svd_best_val = svd_val_scores[best_k_svd]
    svd_best_test = svd_test_scores[best_k_svd]
    als_best_val = best_overall_config[3]
    als_best_test = best_overall_config[4]
    
    print(f"\n{'Method':<15} {'k':<8} {'Val Acc':<12} {'Test Acc':<12}")
    print("-"*50)
    print(f"{'SVD':<15} {best_k_svd:<8} {svd_best_val:<12.4f} {svd_best_test:<12.4f}")
    print(f"{'ALS':<15} {best_k_overall:<8} {als_best_val:<12.4f} {als_best_test:<12.4f}")
    val_diff = als_best_val - svd_best_val
    test_diff = als_best_test - svd_best_test
    print(f"\nDifference: Val={val_diff:+.4f}, Test={test_diff:+.4f}")
    
    print("\n[Creating comparison plot...]")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: ALS training curves
    ax1.plot(range(1, len(losses)+1), losses, label="Train Loss", color="tab:blue", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Training Loss", color="tab:blue", fontsize=11)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(range(1, len(val_accs)+1), val_accs, label="Val Accuracy", 
                   color="tab:orange", linewidth=2, linestyle='--')
    ax1_twin.set_ylabel("Validation Accuracy", color="tab:orange", fontsize=11)
    ax1_twin.tick_params(axis='y', labelcolor='tab:orange')
    ax1_twin.legend(loc='upper right')
    
    # Right: Performance comparison
    methods = ['SVD', 'ALS']
    val_accs_plot = [svd_best_val, als_best_val]
    test_accs_plot = [svd_best_test, als_best_test]
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, val_accs_plot, width, label='Val Acc', 
            color='#3498db', alpha=0.8)
    bars2 = ax2.bar(x + width/2, test_accs_plot, width, label='Test Acc', 
            color='#e74c3c', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Method', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([min(min(val_accs_plot), min(test_accs_plot)) - 0.02, 
                  max(max(val_accs_plot), max(test_accs_plot)) + 0.02])
    
    plt.tight_layout()
    comparison_plot_name = f"mf_comparison_{student_id}.png"
    plt.savefig(comparison_plot_name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {comparison_plot_name}")

if __name__ == "__main__":
    main()