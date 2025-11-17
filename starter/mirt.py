# MLA Fall 2025 - Hanoi University
# Part B: Multi-Dimensional Item Response Theory (MIRT)
# Academic Integrity Declaration:
# This implementation extends the baseline IRT model to incorporate multi-dimensional
# student abilities across different subject areas.


import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
)


def sigmoid(x):
    """Apply sigmoid function with numerical stability.
    
    Formula: σ(x) = 1 / (1 + exp(-x))
    """
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def load_subject_mapping(data_path="./data"):
    """Load and process subject metadata for questions.
    
    Returns:
        question_subjects: dict mapping question_id -> list of subject_ids
        major_subjects: list of major subject categories
    """
    # Load question metadata
    df = pd.read_csv(f"{data_path}/question_meta.csv")
    
    question_subjects = {}
    for _, row in df.iterrows():
        q_id = row['question_id']
        subjects = json.loads(row['subject_id']) if isinstance(row['subject_id'], str) else row['subject_id']
        question_subjects[q_id] = subjects
    
    # Define major subject categories (top-level subjects)
    # Based on the metadata structure, we'll use major categories
    major_subjects = [
        0,   # Maths (general)
        1,   # Number
        17,  # Algebra
        39,  # Ratio and Proportion
        68,  # Geometry
        101, # Probability
        148, # Statistics
    ]
    
    return question_subjects, major_subjects


def create_discrimination_matrix(question_subjects, major_subjects, num_questions):
    """Create discrimination matrix A where A[j,k] indicates if question j 
    involves subject k.
    
    Args:
        question_subjects: dict mapping question_id -> list of subject_ids
        major_subjects: list of major subject category IDs
        num_questions: total number of questions
    
    Returns:
        A: numpy array of shape (num_questions, num_dimensions)
           A[j,k] = 1 if question j involves subject k, else 0
    """
    num_dimensions = len(major_subjects)
    A = np.zeros((num_questions, num_dimensions))
    
    for q_id, subjects in question_subjects.items():
        if q_id < num_questions:  # Ensure question ID is valid
            for k, major_subj in enumerate(major_subjects):
                if major_subj in subjects:
                    A[q_id, k] = 1.0
    
    # Normalize: if a question has no major subjects, assign to general (dim 0)
    for j in range(num_questions):
        if A[j].sum() == 0:
            A[j, 0] = 1.0
    
    # L2 normalize each row to avoid scale issues
    row_sums = A.sum(axis=1, keepdims=True)
    A = A / (row_sums + 1e-10)
    
    return A


def neg_log_likelihood_mirt(data, theta, beta, A):
    """Compute the negative log-likelihood for MIRT.
    
    MIRT Model:
        P(c_ij = 1 | θ_i, β_j, A_j) = σ(Σ_k A_jk * θ_ik - β_j)
    
    Args:
        data: dict with user_id, question_id, is_correct
        theta: matrix of shape (num_users, num_dimensions) - student abilities
        beta: vector of shape (num_questions,) - question difficulties
        A: matrix of shape (num_questions, num_dimensions) - discrimination
    
    Returns:
        negative log-likelihood (scalar)
    """
    log_lklihood = 0.0
    
    for i in range(len(data["user_id"])):
        u = data["user_id"][i]
        q = data["question_id"][i]
        c = data["is_correct"][i]
        
        # Compute MIRT prediction: Σ_k A_jk * θ_ik - β_j
        ability_weighted = np.dot(A[q], theta[u])  # Σ_k A_jk * θ_ik
        x = ability_weighted - beta[q]
        p = sigmoid(x)
        
        # Add log-likelihood contribution
        if c == 1:
            log_lklihood += np.log(p + 1e-10)
        else:
            log_lklihood += np.log(1 - p + 1e-10)
    
    return -log_lklihood


def update_theta_beta_mirt(data, lr, theta, beta, A):
    """Update theta and beta using gradient ascent for MIRT.
    
    Gradients:
        ∂L/∂θ_ik = Σ_j A_jk * (c_ij - p_ij)
        ∂L/∂β_j = Σ_i (p_ij - c_ij)
    
    Args:
        data: dict with user_id, question_id, is_correct
        lr: learning rate
        theta: matrix (num_users, num_dimensions)
        beta: vector (num_questions,)
        A: discrimination matrix (num_questions, num_dimensions)
    
    Returns:
        updated theta, beta
    """
    num_users, num_dimensions = theta.shape
    num_questions = len(beta)
    
    # Initialize gradients
    grad_theta = np.zeros_like(theta)
    grad_beta = np.zeros_like(beta)
    
    # Compute gradients for all observations
    for i in range(len(data["user_id"])):
        u = data["user_id"][i]
        q = data["question_id"][i]
        c = data["is_correct"][i]
        
        # Forward pass
        ability_weighted = np.dot(A[q], theta[u])
        x = ability_weighted - beta[q]
        p = sigmoid(x)
        
        # Error term
        error = c - p
        
        # Gradient for theta: ∂L/∂θ_ik = A_jk * error
        grad_theta[u] += A[q] * error
        
        # Gradient for beta: ∂L/∂β_j = -error
        grad_beta[q] -= error
    
    # Gradient ascent update
    new_theta = theta + lr * grad_theta
    new_beta = beta + lr * grad_beta
    
    return new_theta, new_beta


def evaluate_mirt(data, theta, beta, A):
    """Evaluate MIRT model accuracy.
    
    Args:
        data: dict with user_id, question_id, is_correct
        theta: student abilities matrix
        beta: question difficulties
        A: discrimination matrix
    
    Returns:
        accuracy (float)
    """
    pred = []
    for i in range(len(data["question_id"])):
        u = data["user_id"][i]
        q = data["question_id"][i]
        
        ability_weighted = np.dot(A[q], theta[u])
        x = ability_weighted - beta[q]
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def train_mirt(data, val_data, A, lr, iterations, num_dimensions):
    """Train Multi-Dimensional IRT model.
    
    Args:
        data: training data dict
        val_data: validation data dict
        A: discrimination matrix (num_questions, num_dimensions)
        lr: learning rate
        iterations: number of training iterations
        num_dimensions: number of ability dimensions
    
    Returns:
        theta: trained student abilities
        beta: trained question difficulties
        neg_lld_train_lst: list of training losses
        val_acc_lst: list of validation accuracies
    """
    # Initialize parameters
    num_users = max(data["user_id"]) + 1
    num_questions = max(data["question_id"]) + 1
    
    # Initialize theta as (num_users, num_dimensions) matrix
    theta = np.zeros((num_users, num_dimensions))
    beta = np.zeros(num_questions)
    
    # Track metrics
    neg_lld_train_lst = []
    val_acc_lst = []
    
    print("\n" + "="*80)
    print("Training Multi-Dimensional IRT (MIRT)")
    print("="*80)
    print(f"Number of dimensions: {num_dimensions}")
    print(f"Number of users: {num_users}")
    print(f"Number of questions: {num_questions}")
    print(f"Learning rate: {lr}")
    print(f"Iterations: {iterations}")
    print("-"*80)
    
    for iteration in range(iterations):
        # Compute metrics
        neg_lld = neg_log_likelihood_mirt(data, theta, beta, A)
        val_score = evaluate_mirt(val_data, theta, beta, A)
        
        neg_lld_train_lst.append(neg_lld)
        val_acc_lst.append(val_score)
        
        # Print progress
        if (iteration + 1) % 5 == 0 or iteration == 0:
            print(f"Iter {iteration+1:3d}/{iterations} | NLLK: {neg_lld:8.4f} | Val Acc: {val_score:.4f}")
        
        # Update parameters
        theta, beta = update_theta_beta_mirt(data, lr, theta, beta, A)
    
    print("-"*80)
    print("Training completed!")
    print("="*80 + "\n")
    
    return theta, beta, neg_lld_train_lst, val_acc_lst


def plot_mirt_comparison(baseline_acc, mirt_acc, baseline_loss, mirt_loss, 
                         baseline_val, mirt_val, student_id=""):
    """Plot comparison between baseline IRT and MIRT.
    
    Creates a 2x2 subplot showing:
    1. Validation accuracy comparison
    2. Training loss comparison
    3. Final accuracy bar chart
    4. Improvement metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    iterations = len(baseline_val)
    epochs = np.arange(1, iterations + 1)
    
    # Plot 1: Validation Accuracy
    axes[0, 0].plot(epochs, baseline_val, 'b-', linewidth=2, label='Baseline IRT')
    axes[0, 0].plot(epochs, mirt_val, 'r-', linewidth=2, label='MIRT')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Validation Accuracy')
    axes[0, 0].set_title('Validation Accuracy: MIRT vs Baseline IRT')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Training Loss (NLLK)
    axes[0, 1].plot(epochs, baseline_loss, 'b-', linewidth=2, label='Baseline IRT')
    axes[0, 1].plot(epochs, mirt_loss, 'r-', linewidth=2, label='MIRT')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Negative Log-Likelihood')
    axes[0, 1].set_title('Training Loss: MIRT vs Baseline IRT')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Final Accuracy Comparison
    models = ['Baseline IRT', 'MIRT']
    final_accs = [baseline_acc, mirt_acc]
    colors = ['#3498db', '#e74c3c']
    
    bars = axes[1, 0].bar(models, final_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    axes[1, 0].set_ylabel('Test Accuracy')
    axes[1, 0].set_title('Final Test Accuracy Comparison')
    axes[1, 0].set_ylim([min(final_accs) - 0.02, max(final_accs) + 0.02])
    
    # Add value labels on bars
    for bar, acc in zip(bars, final_accs):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.4f}',
                       ha='center', va='bottom', fontweight='bold')
    
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Improvement Metrics
    improvement = mirt_acc - baseline_acc
    improvement_pct = (improvement / baseline_acc) * 100
    
    axes[1, 1].axis('off')
    
    metrics_text = f"""
    Performance Comparison
    {'='*35}
    
    Baseline IRT Accuracy:  {baseline_acc:.4f}
    MIRT Accuracy:          {mirt_acc:.4f}
    
    Absolute Improvement:   {improvement:+.4f}
    Relative Improvement:   {improvement_pct:+.2f}%
    
    {'='*35}
    
    Final Validation Accuracies:
    Baseline: {baseline_val[-1]:.4f}
    MIRT:     {mirt_val[-1]:.4f}
    
    """
    
    axes[1, 1].text(0.1, 0.5, metrics_text, 
                   fontsize=11, 
                   family='monospace',
                   verticalalignment='center')
    
    plt.suptitle(f'MIRT vs Baseline IRT Comparison (Student ID: {student_id})', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    filename = f'mirt_comparison_{student_id}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Comparison plot saved as '{filename}'")


def plot_ability_heatmap(theta, major_subjects, subject_names, student_id="", 
                        sample_users=30):
    """Plot heatmap of student abilities across different subject dimensions.
    
    Args:
        theta: student ability matrix (num_users, num_dimensions)
        major_subjects: list of subject IDs
        subject_names: dict mapping subject_id -> name
        student_id: for filename
        sample_users: number of users to display
    """
    # Sample users for visualization
    num_users = theta.shape[0]
    if num_users > sample_users:
        user_indices = np.random.choice(num_users, sample_users, replace=False)
        theta_sample = theta[user_indices]
    else:
        theta_sample = theta
        user_indices = np.arange(num_users)
    
    # Create subject labels
    subject_labels = [subject_names.get(subj, f"Subject {subj}") for subj in major_subjects]
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(theta_sample, 
                cmap='RdYlGn', 
                center=0,
                cbar_kws={'label': 'Ability Level'},
                xticklabels=subject_labels,
                yticklabels=[f'User {i}' for i in user_indices],
                linewidths=0.5,
                linecolor='gray')
    
    plt.title(f'Student Abilities Across Subject Dimensions (MIRT)\n(Student ID: {student_id})', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Subject Dimension', fontsize=12, fontweight='bold')
    plt.ylabel('Student ID', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    filename = f'mirt_ability_heatmap_{student_id}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Ability heatmap saved as '{filename}'")


def analyze_subject_performance(theta, A, data, major_subjects, subject_names):
    """Analyze performance across different subject dimensions.
    
    Returns statistics about which subjects students perform best/worst in.
    """
    num_dimensions = theta.shape[1]
    
    print("\n" + "="*80)
    print("Subject-wise Performance Analysis")
    print("="*80)
    
    # Average ability per dimension
    avg_abilities = theta.mean(axis=0)
    std_abilities = theta.std(axis=0)
    
    for k, subj_id in enumerate(major_subjects):
        subj_name = subject_names.get(subj_id, f"Subject {subj_id}")
        print(f"{subj_name:30s} | Mean: {avg_abilities[k]:7.4f} | Std: {std_abilities[k]:7.4f}")
    
    print("="*80 + "\n")
    
    return avg_abilities, std_abilities


def main():
    """Main function to train and evaluate MIRT model."""
    
    # Load data
    print("Loading data...")
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")
    
    print(f"[OK] Training samples: {len(train_data['user_id'])}")
    print(f"[OK] Validation samples: {len(val_data['user_id'])}")
    print(f"[OK] Test samples: {len(test_data['user_id'])}")
    
    # Load subject metadata
    print("\nLoading subject metadata...")
    question_subjects, major_subjects = load_subject_mapping("./data")
    
    # Load subject names
    subject_df = pd.read_csv("./data/subject_meta.csv")
    subject_names = dict(zip(subject_df['subject_id'], subject_df['name']))
    
    print(f"[OK] Loaded {len(question_subjects)} questions with subject info")
    print(f"[OK] Using {len(major_subjects)} major subject dimensions:")
    for subj_id in major_subjects:
        print(f"  - {subj_id}: {subject_names.get(subj_id, 'Unknown')}")
    
    # Create discrimination matrix
    num_questions = max(train_data["question_id"]) + 1
    A = create_discrimination_matrix(question_subjects, major_subjects, num_questions)
    print(f"\n[OK] Created discrimination matrix of shape {A.shape}")
    
    # Student ID
    STUDENT_ID = "2201040173" 
    
    # Hyperparameters
    LR = 0.01
    ITERATIONS = 50
    NUM_DIMENSIONS = len(major_subjects)
    
    #############################################################################
    # Train MIRT
    #############################################################################
    
    theta, beta, mirt_loss, mirt_val = train_mirt(
        train_data, val_data, A, LR, ITERATIONS, NUM_DIMENSIONS
    )
    
    # Evaluate on test set
    mirt_test_acc = evaluate_mirt(test_data, theta, beta, A)
    
    print("="*80)
    print("MIRT RESULTS")
    print("="*80)
    print(f"Final Validation Accuracy: {mirt_val[-1]:.4f}")
    print(f"Final Test Accuracy:       {mirt_test_acc:.4f}")
    print("="*80 + "\n")
    
    #############################################################################
    # Train Baseline IRT for comparison
    #############################################################################
    
    print("Training baseline IRT for comparison...")
    from item_response import irt, evaluate as irt_evaluate
    
    theta_baseline, beta_baseline, baseline_loss, baseline_val = irt(
        train_data, val_data, LR, ITERATIONS
    )
    
    baseline_test_acc = irt_evaluate(test_data, theta_baseline, beta_baseline)
    
    print("="*80)
    print("BASELINE IRT RESULTS")
    print("="*80)
    print(f"Final Validation Accuracy: {baseline_val[-1]:.4f}")
    print(f"Final Test Accuracy:       {baseline_test_acc:.4f}")
    print("="*80 + "\n")
    
    #############################################################################
    # Comparison and Visualization
    #############################################################################
    
    print("Generating comparison plots...")
    plot_mirt_comparison(
        baseline_test_acc, mirt_test_acc,
        baseline_loss, mirt_loss,
        baseline_val, mirt_val,
        STUDENT_ID
    )
    
    # Plot ability heatmap
    print("\nGenerating ability heatmap...")
    plot_ability_heatmap(theta, major_subjects, subject_names, STUDENT_ID)
    
    # Analyze subject performance
    analyze_subject_performance(theta, A, train_data, major_subjects, subject_names)
    
    #############################################################################
    # Summary
    #############################################################################
    
    improvement = mirt_test_acc - baseline_test_acc
    improvement_pct = (improvement / baseline_test_acc) * 100
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Baseline IRT Test Accuracy:  {baseline_test_acc:.4f}")
    print(f"MIRT Test Accuracy:          {mirt_test_acc:.4f}")
    print(f"Absolute Improvement:        {improvement:+.4f}")
    print(f"Relative Improvement:        {improvement_pct:+.2f}%")
    print("="*80)
    
    print("\n[OK] All tasks completed successfully!")
    print("\nGenerated files:")
    print(f"  1. mirt_comparison_{STUDENT_ID}.png")
    print(f"  2. mirt_ability_heatmap_{STUDENT_ID}.png")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

