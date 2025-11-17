# MLA Fall 2025 - Hanoi University
# Academic Integrity Declaration:
# I, 2201040173 (Dam Thanh Thuy), declare that this code is my own original work.
# I have not copied or adapted code from any external repositories or previous years.
# Any sources or libraries used are explicitly cited below.

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
    evaluate,
)

# Import base models
from item_response import irt, evaluate as irt_evaluate
from matrix_factorization import als
from knn import item_knn_predict_hanu


MF_CONFIG_PATH = Path(__file__).resolve().parent / "mf_best_config.json"
DEFAULT_MF_CONFIG = {
    "k": 50,
    "lr": 0.01,
    "lambda_": 0.01,
    "iterations": 10,
}


def load_best_mf_config(path: Path = MF_CONFIG_PATH):
    """Load best MF hyperparameters saved by matrix_factorization.py."""
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            cfg = payload.get("best_config", {})
            required_keys = {"k", "lr", "lambda", "epochs"}
            if required_keys.issubset(cfg.keys()):
                loaded = {
                    "k": int(cfg["k"]),
                    "lr": float(cfg["lr"]),
                    "lambda_": float(cfg["lambda"]),
                    "iterations": int(cfg["epochs"]),
                }
                print(
                    "[OK] Loaded MF hyperparameters from "
                    f"{path.name}: k={loaded['k']}, lr={loaded['lr']}, "
                    f"lambda={loaded['lambda_']}, epochs={loaded['iterations']}"
                )
                return loaded
            else:
                print(
                    "[WARN] mf_best_config.json missing required keys. "
                    "Falling back to default MF config."
                )
        except (OSError, ValueError, json.JSONDecodeError) as err:
            print(
                f"[WARN] Could not parse {path.name} ({err}). "
                "Falling back to default MF config."
            )
    else:
        print(
            "[WARN] mf_best_config.json not found. "
            "Run matrix_factorization.py first to generate it."
        )
    return DEFAULT_MF_CONFIG.copy()


def create_bootstrap_sample(data, seed=None):
    """
    Create a bootstrap sample (random sample with replacement) from the data.
    
    Args:
        data: dict with keys 'user_id', 'question_id', 'is_correct'
        seed: random seed for reproducibility
    
    Returns:
        bootstrap_data: dict with same structure as input, sampled with replacement
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = len(data["user_id"])
    # Sample indices with replacement
    bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
    
    # Create bootstrap sample
    bootstrap_data = {
        "user_id": [data["user_id"][i] for i in bootstrap_indices],
        "question_id": [data["question_id"][i] for i in bootstrap_indices],
        "is_correct": [data["is_correct"][i] for i in bootstrap_indices],
    }
    
    return bootstrap_data


def train_irt_model(train_data, val_data, lr=0.01, iterations=25):
    """
    Train an IRT model on the given data.
    
    Args:
        train_data: training data dict
        val_data: validation data dict
        lr: learning rate
        iterations: number of training iterations
    
    Returns:
        theta: student abilities
        beta: question difficulties
    """
    print(f"  Training IRT model (lr={lr}, iter={iterations})...")
    theta, beta, _, _ = irt(train_data, val_data, lr, iterations)
    val_acc = irt_evaluate(val_data, theta, beta)
    print(f"    -> IRT validation accuracy: {val_acc:.4f}")
    return theta, beta


def create_bootstrap_matrix(train_data, train_matrix, seed=None):
    """
    Create a bootstrap matrix from bootstrap data.
    
    Args:
        train_data: original training data dict
        train_matrix: original training matrix
        seed: random seed for bootstrap sampling
    
    Returns:
        bootstrap_matrix: matrix created from bootstrap sample
    """
    bootstrap_data = create_bootstrap_sample(train_data, seed=seed)
    
    # Create matrix from bootstrap data
    num_users = max(train_data["user_id"]) + 1
    num_questions = max(train_data["question_id"]) + 1
    bootstrap_matrix = np.full((num_users, num_questions), np.nan)
    
    for i in range(len(bootstrap_data["user_id"])):
        u = bootstrap_data["user_id"][i]
        q = bootstrap_data["question_id"][i]
        c = bootstrap_data["is_correct"][i]
        bootstrap_matrix[u, q] = float(c)
    
    return bootstrap_matrix

def train_knn_model(train_matrix, val_data, k=21):
    """
    Train a KNN model (item-based) on the given data.
    
    Args:
        train_matrix: sparse training matrix (users x questions)
        val_data: validation data dict
        k: number of neighbors
    
    Returns:
        filled_matrix: imputed matrix with predictions
    """
    print(f"  Training Item-based KNN model (k={k})...")
    
    # Transpose for item-based similarity
    matrix_t = train_matrix.T
    imputer = KNNImputer(n_neighbors=k)
    mat_t = imputer.fit_transform(matrix_t)
    filled_matrix = mat_t.T
    
    val_acc = sparse_matrix_evaluate(val_data, filled_matrix)
    print(f"    -> KNN validation accuracy: {val_acc:.4f}")
    return filled_matrix


def train_mf_model(train_data, val_data, k=50, lr=0.01, iterations=10, lambda_=0.01):
    """
    Train a Matrix Factorization model using ALS (Alternating Least Squares).
    
    This function wraps the ALS implementation from matrix_factorization.py
    and adapts it for use in ensemble methods.
    
    Args:
        train_data: training data dict with keys 'user_id', 'question_id', 'is_correct'
        val_data: validation data dict with same structure
        k: latent dimension (number of latent factors)
        lr: learning rate for gradient descent
        iterations: number of training epochs
        lambda_: L2 regularization parameter
    
    Returns:
        pred_matrix: prediction matrix (users x questions) with values in [0, 1]
    """
    print(f"  Training Matrix Factorization model (k={k}, lr={lr}, lambda={lambda_}, iter={iterations})...")
    
    # ALS function returns tuple: (pred_matrix, losses, val_accs)
    # We only need the prediction matrix for ensemble, so we unpack and ignore losses/val_accs
    # plot=False to avoid generating plots during ensemble training
    pred_matrix, _, _ = als(
        train_data=train_data, 
        valid_data=val_data, 
        k=k, 
        lr=lr, 
        num_iteration=iterations, 
        lambda_=lambda_, 
        student_id="",  # Empty string since we don't need to save plots
        plot=False  # Disable plotting for ensemble training
    )
    
    # Ensure predictions are in valid probability range [0, 1]
    pred_matrix = np.clip(pred_matrix, 0.0, 1.0)
    
    # Evaluate on validation set
    val_acc = sparse_matrix_evaluate(val_data, pred_matrix)
    print(f"    -> MF validation accuracy: {val_acc:.4f}")
    
    return pred_matrix


def get_predictions_from_matrix(matrix, data):
    """
    Extract predictions from a prediction matrix for given data points.
    
    Args:
        matrix: prediction matrix (users x questions)
        data: dict with 'user_id' and 'question_id'
    
    Returns:
        predictions: array of predictions
    """
    predictions = []
    for i in range(len(data["user_id"])):
        u = data["user_id"][i]
        q = data["question_id"][i]
        predictions.append(matrix[u, q])
    return np.array(predictions)


def get_irt_predictions(theta, beta, data):
    """
    Get predictions from IRT model.
    
    Args:
        theta: student abilities
        beta: question difficulties
        data: dict with 'user_id' and 'question_id'
    
    Returns:
        predictions: array of probability predictions
    """
    def sigmoid(x):
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)),
                       np.exp(x) / (1 + np.exp(x)))
    
    predictions = []
    for i in range(len(data["user_id"])):
        u = data["user_id"][i]
        q = data["question_id"][i]
        x = theta[u] - beta[q]
        p = sigmoid(x)
        predictions.append(p)
    
    return np.array(predictions)


def bagging_ensemble(train_data, train_matrix, val_data, test_data, 
                     n_models=3, random_seeds=None):
    """
    Implement bagging ensemble with 3 base models.
    
    Each model is trained on a bootstrap sample (random sample with replacement).
    Final prediction is the average of all base model predictions.
    
    Args:
        train_data: training data dict
        train_matrix: sparse training matrix
        val_data: validation data dict
        test_data: test data dict
        n_models: number of base models (default: 3)
        random_seeds: list of random seeds for bootstrap sampling
    
    Returns:
        ensemble_val_acc: validation accuracy of ensemble
        ensemble_test_acc: test accuracy of ensemble
        individual_accs: list of individual model accuracies
    """
    if random_seeds is None:
        random_seeds = [42, 123, 456]
    
    print("\n" + "="*70)
    print("BAGGING ENSEMBLE - Training Base Models")
    print("="*70)
    
    # Store predictions from each model
    val_predictions_list = []
    test_predictions_list = []
    individual_val_accs = []
    
    mf_params = load_best_mf_config()

    # Strategy: Use 3 different model types with bootstrap samples
    model_configs = [
        {"type": "irt", "params": {"lr": 0.01, "iterations": 25}},
        {"type": "knn", "params": {"k": 21}},
        {"type": "mf", "params": mf_params},
    ]
    
    for i in range(n_models):
        print(f"\n--- Base Model {i+1}/{n_models} ---")
        seed = random_seeds[i]
        
        # Create bootstrap sample
        print(f"Creating bootstrap sample (seed={seed})...")
        bootstrap_data = create_bootstrap_sample(train_data, seed=seed)
        print(f"  Bootstrap sample size: {len(bootstrap_data['user_id'])}")
        
        # Count unique samples (some may be duplicated due to replacement)
        unique_indices = len(set(zip(bootstrap_data["user_id"], 
                                     bootstrap_data["question_id"])))
        print(f"  Unique samples: {unique_indices}")
        
        model_config = model_configs[i]
        model_type = model_config["type"]
        params = model_config["params"]
        
        # Train model based on type
        if model_type == "irt":
            # Train IRT model
            theta, beta = train_irt_model(bootstrap_data, val_data, **params)
            
            # Get predictions
            val_preds = get_irt_predictions(theta, beta, val_data)
            test_preds = get_irt_predictions(theta, beta, test_data)
            
        elif model_type == "knn":
            # Create bootstrap matrix from bootstrap data
            bootstrap_matrix = create_bootstrap_matrix(train_data, train_matrix, seed=seed)
            filled_matrix = train_knn_model(bootstrap_matrix, val_data, **params)
            
            val_preds = get_predictions_from_matrix(filled_matrix, val_data)
            test_preds = get_predictions_from_matrix(filled_matrix, test_data)
            
        elif model_type == "mf":
            # Train Matrix Factorization
            pred_matrix = train_mf_model(bootstrap_data, val_data, **params)
            
            val_preds = get_predictions_from_matrix(pred_matrix, val_data)
            test_preds = get_predictions_from_matrix(pred_matrix, test_data)
        
        # Store predictions
        val_predictions_list.append(val_preds)
        test_predictions_list.append(test_preds)
        
        # Calculate individual accuracy
        val_acc = evaluate(val_data, val_preds)
        individual_val_accs.append(val_acc)
        print(f"  -> Model {i+1} validation accuracy: {val_acc:.4f}")
    
    print("\n" + "="*70)
    print("ENSEMBLE PREDICTIONS - Averaging")
    print("="*70)
    
    # Average predictions from all models
    ensemble_val_preds = np.mean(val_predictions_list, axis=0)
    ensemble_test_preds = np.mean(test_predictions_list, axis=0)
    
    # Calculate ensemble accuracy
    ensemble_val_acc = evaluate(val_data, ensemble_val_preds)
    ensemble_test_acc = evaluate(test_data, ensemble_test_preds)
    
    print(f"\nIndividual Model Accuracies:")
    for i, acc in enumerate(individual_val_accs):
        print(f"  Model {i+1}: {acc:.4f}")
    print(f"\nEnsemble Validation Accuracy: {ensemble_val_acc:.4f}")
    print(f"Ensemble Test Accuracy:       {ensemble_test_acc:.4f}")
    
    # Calculate improvement
    avg_individual = np.mean(individual_val_accs)
    improvement = ensemble_val_acc - avg_individual
    print(f"\nImprovement over average: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    return ensemble_val_acc, ensemble_test_acc, individual_val_accs


def plot_ensemble_results(individual_accs, ensemble_acc, save_path="ensemble_results.png"):
    """
    Plot comparison of individual models vs ensemble.
    
    Args:
        individual_accs: list of individual model validation accuracies
        ensemble_acc: ensemble validation accuracy
        save_path: path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_models = len(individual_accs)
    x = np.arange(n_models + 1)
    
    # Plot individual models
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for i, acc in enumerate(individual_accs):
        ax.bar(i, acc, color=colors[i], alpha=0.7, 
               label=f'Model {i+1}: {acc:.4f}')
    
    # Plot ensemble
    ax.bar(n_models, ensemble_acc, color='#f39c12', alpha=0.9, 
           label=f'Ensemble: {ensemble_acc:.4f}', edgecolor='black', linewidth=2)
    
    # Add horizontal line for average individual performance
    avg_individual = np.mean(individual_accs)
    ax.axhline(y=avg_individual, color='red', linestyle='--', linewidth=1.5,
               label=f'Avg Individual: {avg_individual:.4f}')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Bagging Ensemble: Individual Models vs Ensemble', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Model 1\n(IRT)', 'Model 2\n(KNN)', 'Model 3\n(MF)', 'Ensemble'])
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([min(individual_accs) - 0.01, max(ensemble_acc, max(individual_accs)) + 0.01])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {save_path}")


def main():   
    # Load data
    print("\nLoading data...")
    train_data = load_train_csv("./data")
    train_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")
    
    print(f"Training samples: {len(train_data['user_id'])}")
    print(f"Validation samples: {len(val_data['user_id'])}")
    print(f"Test samples: {len(test_data['user_id'])}")
    print(f"Matrix shape: {train_matrix.shape}")
    
    # Run bagging ensemble
    ensemble_val_acc, ensemble_test_acc, individual_accs = bagging_ensemble(
        train_data=train_data,
        train_matrix=train_matrix,
        val_data=val_data,
        test_data=test_data,
        n_models=3,
        random_seeds=[42, 123, 456]
    )
    
    # Plot results
    plot_ensemble_results(individual_accs, ensemble_val_acc)
    # Final summary
    print(f"Ensemble Validation Accuracy: {ensemble_val_acc:.4f}")
    print(f"Ensemble Test Accuracy:       {ensemble_test_acc:.4f}")
    print("This bagging ensemble uses 3 different base models:")
    print("  1. IRT (Item Response Theory)")
    print("  2. Item-based KNN (k-Nearest Neighbors)")
    print("  3. Matrix Factorization (ALS)")

if __name__ == "__main__":
    main()
