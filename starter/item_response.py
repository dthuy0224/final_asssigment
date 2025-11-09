# MLA Fall 2025 - Hanoi University
# Academic Integrity Declaration:
# I, [TranHoaiNam] ([2201040120]), declare that this code is my own original work.
# I have not copied or adapted code from any external repositories or previous years.
# Any sources or libraries used are explicitly cited below.

from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """Apply sigmoid function with numerical stability.
    
    Uses different formulations for positive and negative x
    to avoid overflow in np.exp().
    
    Formula: σ(x) = 1 / (1 + exp(-x))
    """
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    The log-likelihood is:
    log p(C|θ,β) = Σ [c_ij(θ_i - β_j) - log(1 + exp(θ_i - β_j))]
    
    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param theta: Vector of student abilities
    :param beta: Vector of question difficulties
    :return: float (negative log-likelihood)
    """
    log_lklihood = 0.0
    
    for i in range(len(data["user_id"])):
        u = data["user_id"][i]
        q = data["question_id"][i]
        c = data["is_correct"][i]

        # Calculate x = theta_i - beta_j
        x = theta[u] - beta[q]
        p = sigmoid(x)

        # Add log-likelihood contribution
        # log p(c_ij | θ_i, β_j) = c_ij * log(p) + (1-c_ij) * log(1-p)
        if c == 1:
            log_lklihood += np.log(p + 1e-10)  # Add small epsilon to avoid log(0)
        else:  # c == 0
            log_lklihood += np.log(1 - p + 1e-10)
            
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient ascent.

    Gradients:
    ∂ log p / ∂θ_i = Σ_j [c_ij - σ(θ_i - β_j)]
    ∂ log p / ∂β_j = Σ_i [σ(θ_i - β_j) - c_ij]
    
    Since we're maximizing log-likelihood, we ADD the gradient.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param lr: float (learning rate)
    :param theta: Vector of student abilities
    :param beta: Vector of question difficulties
    :return: tuple of vectors (updated theta, updated beta)
    """
    # Initialize gradients
    grad_theta = np.zeros_like(theta)
    grad_beta = np.zeros_like(beta)
    
    # Compute gradients for all observations
    for i in range(len(data["user_id"])):
        u = data["user_id"][i]
        q = data["question_id"][i]
        c = data["is_correct"][i]

        x = theta[u] - beta[q]
        p = sigmoid(x)
        
        # Gradient term: (c_ij - p(c_ij=1|...))
        error = c - p

        # Accumulate gradients
        grad_theta[u] += error
        grad_beta[q] -= error  # Note: negative because ∂/∂β_j = -(c_ij - p)

    # Perform gradient ascent update (maximizing log-likelihood)
    new_theta = theta + lr * grad_theta
    new_beta = beta + lr * grad_beta
    
    return new_theta, new_beta


def irt(data, val_data, lr, iterations):
    """Train IRT model using alternating gradient ascent.

    :param data: Training data dictionary
    :param val_data: Validation data dictionary
    :param lr: float (learning rate)
    :param iterations: int (number of iterations)
    :return: (theta, beta, neg_lld_train_lst, val_acc_lst)
    """
    # Initialize parameters
    num_users = max(data["user_id"]) + 1
    num_questions = max(data["question_id"]) + 1
    
    # Initialize to zeros (or small random values)
    theta = np.zeros(num_users)
    beta = np.zeros(num_questions)

    # Track metrics for plotting
    neg_lld_train_lst = []
    val_acc_lst = []

    print("\nTraining IRT Model...")
    print("-" * 70)
    
    for i in range(iterations):
        # Compute training loss and validation accuracy
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        val_score = evaluate(data=val_data, theta=theta, beta=beta)
        
        neg_lld_train_lst.append(neg_lld)
        val_acc_lst.append(val_score)
        
        # Print progress every 5 iterations
        if (i + 1) % 5 == 0 or i == 0:
            print(f"Iteration {i+1:3d}/{iterations} | NLLK: {neg_lld:8.4f} | Val Acc: {val_score:.4f}")
        
        # Update parameters using gradient ascent
        theta, beta = update_theta_beta(data, lr, theta, beta)

    print("-" * 70)
    print("Training completed!\n")
    
    return theta, beta, neg_lld_train_lst, val_acc_lst


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    
    Prediction rule: predict 1 if p(c_ij=1) >= 0.5, else predict 0
    
    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param theta: Vector of student abilities
    :param beta: Vector of question difficulties
    :return: float (accuracy)
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = theta[u] - beta[q]
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
        
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def plot_irt_curves(neg_lld_train_lst, val_acc_lst, student_id):
    """Plot training NLLK and validation accuracy vs iteration.
    
    This function creates a dual-axis plot showing:
    - Training negative log-likelihood (left axis, red)
    - Validation accuracy (right axis, blue)
    
    :param neg_lld_train_lst: List of NLLK values
    :param val_acc_lst: List of validation accuracies
    :param student_id: Student ID for filename
    """
    iterations = len(neg_lld_train_lst)
    epochs = np.arange(1, iterations + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot NLLK on primary axis (ax1)
    color = 'tab:red'
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Training Negative Log-Likelihood', color=color, fontsize=12)
    ax1.plot(epochs, neg_lld_train_lst, color=color, linewidth=2, label='Training NLLK')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Create secondary axis (ax2) for validation accuracy
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation Accuracy', color=color, fontsize=12)
    ax2.plot(epochs, val_acc_lst, color=color, linewidth=2, label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and save
    plt.title(f'IRT Model: Training Loss and Validation Accuracy\n(Student ID: {student_id})', 
              fontsize=14, pad=20)
    fig.tight_layout()
    
    filename = f'irt_curves_{student_id}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves plot saved as '{filename}'")


def plot_probability_curves(theta, beta, student_id, q_ids):
    """Plot p(c_ij=1) as a function of theta_i for selected questions.
    
    This creates Item Characteristic Curves (ICC) showing how the
    probability of answering correctly varies with student ability.
    
    :param theta: Vector of student abilities
    :param beta: Vector of question difficulties
    :param student_id: Student ID for filename
    :param q_ids: List of question IDs to plot
    """
    # Create a range of theta values (student ability) for plotting
    theta_range = np.linspace(-4, 4, 200)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for idx, q_id in enumerate(q_ids):
        difficulty = beta[q_id]
        # p(c_ij=1 | theta_i, beta_j) = sigmoid(theta_i - beta_j)
        probability = sigmoid(theta_range - difficulty)
        
        color = colors[idx % len(colors)]
        ax.plot(theta_range, probability, 
                color=color, 
                linewidth=2.5,
                label=f'Question {q_id} (β={difficulty:.3f})')
        
        # Mark the inflection point (theta = beta) with vertical dashed line
        ax.axvline(x=difficulty, color=color, linestyle='--', alpha=0.3, linewidth=1)
        
        # Mark the inflection point
        ax.plot(difficulty, 0.5, 'o', color=color, markersize=8, 
                markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlabel('Student Ability (θ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('P(Correct Answer | θ, β)', fontsize=12, fontweight='bold')
    ax.set_title(f'IRT Item Characteristic Curves\n(Student ID: {student_id})', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([theta_range[0], theta_range[-1]])
    
    # Add horizontal line at p=0.5
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    filename = f'irt_probability_{student_id}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Probability curves plot saved as '{filename}'")


def main():
    """Main function to run IRT model training and evaluation."""
    
    # Load data
    print("=" * 70)
    print("PART A - QUESTION 2: ITEM RESPONSE THEORY")
    print("=" * 70)
    print("\nLoading data...")
    
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")
    
    print(f"✓ Training samples: {len(train_data['user_id'])}")
    print(f"✓ Validation samples: {len(val_data['user_id'])}")
    print(f"✓ Test samples: {len(test_data['user_id'])}")
    
    # Student ID for file naming (REPLACE WITH YOUR ACTUAL STUDENT ID)
    STUDENT_ID = "2201040120"
    
    #####################################################################
    # Part (b): Training and Hyperparameter Selection
    #####################################################################
    
    print("\n" + "=" * 70)
    print("HYPERPARAMETER SELECTION")
    print("=" * 70)
    
    # Hyperparameters (YOU SHOULD TUNE THESE)
    LR = 0.01
    ITERATIONS = 50
    
    print(f"Learning Rate: {LR}")
    print(f"Iterations: {ITERATIONS}")
    
    # Train the model
    theta, beta, neg_lld_train_lst, val_acc_lst = irt(
        train_data, val_data, LR, ITERATIONS
    )
    
    #####################################################################
    # Part (c): Evaluation
    #####################################################################
    
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    final_val_acc = val_acc_lst[-1]
    final_test_acc = evaluate(test_data, theta, beta)
    
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Final Test Accuracy:       {final_test_acc:.4f}")
    print("=" * 70)
    
    #####################################################################
    # Part (b): Plot training curves
    #####################################################################
    
    print("\nGenerating plots...")
    plot_irt_curves(neg_lld_train_lst, val_acc_lst, STUDENT_ID)
    
    #####################################################################
    # Part (d): Visualization of Probability Curves
    #####################################################################
    
    # Method 1: Manual selection (uncomment if you prefer)
    # q_j1, q_j2, q_j3 = 10, 100, 500
    
    # Method 2: Automatic selection based on difficulty distribution
    sorted_idx = np.argsort(beta)
    n_questions = len(beta)
    
    q_j1 = sorted_idx[n_questions // 4]        # Easy (25th percentile)
    q_j2 = sorted_idx[n_questions // 2]        # Medium (50th percentile)
    q_j3 = sorted_idx[3 * n_questions // 4]    # Hard (75th percentile)
    
    selected_q_ids = [q_j1, q_j2, q_j3]
    
    print(f"\nSelected questions for visualization:")
    print(f"  Question {q_j1:4d}: β = {beta[q_j1]:7.4f} (Easy - 25th percentile)")
    print(f"  Question {q_j2:4d}: β = {beta[q_j2]:7.4f} (Medium - 50th percentile)")
    print(f"  Question {q_j3:4d}: β = {beta[q_j3]:7.4f} (Hard - 75th percentile)")
    
    plot_probability_curves(theta, beta, STUDENT_ID, selected_q_ids)
    
    #####################################################################
    # Additional Statistics for Report
    #####################################################################
    
    print("\n" + "=" * 70)
    print("MODEL STATISTICS (for your report)")
    print("=" * 70)
    print(f"Number of students:  {len(theta)}")
    print(f"Number of questions: {len(beta)}")
    print(f"\nStudent Ability (θ):")
    print(f"  Mean:  {np.mean(theta):7.4f}")
    print(f"  Std:   {np.std(theta):7.4f}")
    print(f"  Range: [{np.min(theta):7.4f}, {np.max(theta):7.4f}]")
    print(f"\nQuestion Difficulty (β):")
    print(f"  Mean:  {np.mean(beta):7.4f}")
    print(f"  Std:   {np.std(beta):7.4f}")
    print(f"  Range: [{np.min(beta):7.4f}, {np.max(beta):7.4f}]")
    print("=" * 70)
    
    print("\n✓ All tasks completed successfully!")
    print("✓ Please include the generated plots in your report.")
    print("✓ Remember to write the theoretical derivations for Part (a).")
    print("✓ Remember to interpret the probability curves for Part (d).")
    print("\nFiles generated:")
    print(f"  1. irt_curves_{STUDENT_ID}.png")
    print(f"  2. irt_probability_{STUDENT_ID}.png")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()