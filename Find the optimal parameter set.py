import numpy as np

def identify_non_consensus_experts(expert_matrices, threshold=0.8):
    """
    Core algorithm for identifying non-consensus experts
    
    Parameters:
        expert_matrices (list): List of expert matrices [matrix1, matrix2, ...]
        threshold (float): Consensus degree threshold (default: 0.8)
    
    Returns:
        list: Names of non-consensus experts ["Expert 1", ...]
    """
    # Calculate group decision matrix
    group_matrix = np.mean(expert_matrices, axis=0)
    
    non_consensus = []
    
    # Evaluate each expert
    for idx, matrix in enumerate(expert_matrices, 1):
        # Calculate Manhattan distance
        manhattan_dist = np.sum(np.abs(matrix - group_matrix))
        
        # Normalize consensus degree
        n = matrix.shape[0]  # Matrix dimension
        consensus = 1 - manhattan_dist / (n * (n - 1))
        
        # Threshold judgment
        if consensus < threshold:
            non_consensus.append(f"Expert {idx}")
        
        # Diagnostic output (optional)
        print(f"Expert {idx} Consensus: {consensus:.4f}")
    
    return non_consensus

# Usage Example
if __name__ == "__main__":
    # Sample expert matrices (4x4)
    expert_matrices = [
        np.array([[0.5,0.3,0.8,0.6], [0.7,0.5,0.6,0.4], 
                 [0.2,0.4,0.5,0.3], [0.4,0.6,0.7,0.5]]),
        np.array([[0.5,0.7,0.6,0.3], [0.3,0.5,0.5,0.4],
                 [0.4,0.5,0.5,0.7], [0.7,0.6,0.3,0.5]]),
        np.array([[0.5,0.4,0.3,0.9], [0.6,0.5,0.4,0.8],
                 [0.7,0.6,0.5,0.2], [0.1,0.2,0.8,0.5]]),
        np.array([[0.5,0.1,0.9,0.8], [0.9,0.5,0.7,0.4],
                 [0.1,0.3,0.5,0.9], [0.2,0.6,0.1,0.5]])
    ]
    
    # Execute identification
    result = identify_non_consensus_experts(expert_matrices)
    
    # Output results
    print("\nNon-consensus experts:", result if result else "All experts reached consensus")
