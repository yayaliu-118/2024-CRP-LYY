for eta in eta_values:  # First layer: Weight strategy loop
    # Build group decision matrix
    group_matrix = (
        matrix1 * weights[int(eta*8-1)][1] + 
        matrix2 * weights[int(eta*8-1)][0] + 
        matrix3 * weights[int(eta*8-1)][3] + 
        matrix4 * weights[int(eta*8-1)][2]
    )
    
    for delta in delta_values:  # Second layer: Adjustment intensity loop
        # Matrix adjustment calculation
        modified_matrix3 = delta * matrix3 + (1 - delta) * group_matrix
        modified_matrix4 = delta * matrix4 + (1 - delta) * group_matrix
        
        # Three-dimensional metric evaluation
        # Consensus degree calculation
        consensus = 1 - np.sum(np.abs(modified_matrix3 - group_matrix)) / 12
        
        # Modification cost calculation
        cost = 1 - (np.sum(np.abs(matrix3 - modified_matrix3)) + 
                   np.sum(np.abs(matrix4 - modified_matrix4))) / 24
        
        # Consistency calculation
        consistency = 1 - 4 * (
            abs(modified_matrix3[0,1] + modified_matrix3[1,2] - modified_matrix3[0,2] - 0.5 +
            abs(modified_matrix3[1,2] + modified_matrix3[2,3] - modified_matrix3[1,3] - 0.5 +
            abs(modified_matrix3[2,3] + modified_matrix3[3,0] - modified_matrix3[2,0] - 0.5 +
            abs(modified_matrix3[3,0] + modified_matrix3[0,1] - modified_matrix3[3,1] - 0.5 +
            abs(modified_matrix4[0,1] + modified_matrix4[1,2] - modified_matrix4[0,2] - 0.5 +
            abs(modified_matrix4[1,2] + modified_matrix4[2,3] - modified_matrix4[1,3] - 0.5 +
            abs(modified_matrix4[2,3] + modified_matrix4[3,0] - modified_matrix4[2,0] - 0.5 +
            abs(modified_matrix4[3,0] + modified_matrix4[0,1] - modified_matrix4[3,1] - 0.5))
        ) / 24
        
        # Comprehensive metric calculation
        TCL = 0.5 * consensus + 0.5 * (1 - cost)
        
        # Optimal solution recording
        if consistency > best_consistency and TCL > 0.5:
            update_optimal_parameters(eta, delta, consistency, TCL)
