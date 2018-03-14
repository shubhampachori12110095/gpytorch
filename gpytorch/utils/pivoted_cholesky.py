import torch


def pivoted_cholesky(matrix, max_iter, error_tol=1e-5):
    matrix_size = matrix.size(-1)
    matrix_diag = matrix.diag()

    # TODO: This check won't be necessary in PyTorch 0.4
    if isinstance(matrix_diag, torch.autograd.Variable):
        matrix_diag = matrix_diag.data

    error = torch.norm(matrix_diag, 1)
    permutation = matrix_diag.new(matrix_size).long()
    torch.arange(0, matrix_size, out=permutation)

    m = 0
    # TODO: pivoted_cholesky should take tensor_cls and use that here instead
    L = matrix_diag.new(max_iter, matrix_size).zero_()
    while m < max_iter and error > error_tol:
        max_diag_value, max_diag_index = torch.max(matrix_diag[permutation][m:], 0)
        max_diag_index = max_diag_index + m

        pi_m = permutation[m]
        permutation[m] = permutation[max_diag_index][0]
        permutation[max_diag_index] = pi_m

        pi_m = permutation[m]

        L_m = L[m] # Will be all zeros -- should we use torch.zeros?
        L_m[pi_m] = torch.sqrt(max_diag_value)[0]

        row = matrix[pi_m]

        if isinstance(row, torch.autograd.Variable):
            row = row.data

        pi_i = permutation[m + 1:]
        L_m[pi_i] = row[pi_i]
        if m > 0:
            L_prev = L[:m].index_select(1, pi_i)
            L_m[pi_i] -= torch.sum(L[:m, pi_m].unsqueeze(1) * L_prev, dim=0)
        L_m[pi_i] /= L_m[pi_m]

        matrix_diag[pi_i] = matrix_diag[pi_i] - (L_m[pi_i] ** 2)
        L[m] = L_m

        error = torch.sum(matrix_diag[permutation[m + 1:]])
        m = m + 1

    return L[:m, :]


def woodbury_factor(low_rank_mat, shift):
    """
    Given a k-by-n matrix V and a shift \sigma, computes R so that
    R'R = V'(I_k + 1/\sigma VV')^{-1}V
    """
    n = low_rank_mat.size(-1)
    k = low_rank_mat.size(-2)
    shifted_mat = (1 / shift) * low_rank_mat.matmul(low_rank_mat.t())
    shifted_mat = shifted_mat + shifted_mat.new(k).fill_(1).diag()
    # P'P = I_k + 1/\sigma VV'
    # Therefore, V'(I_k + 1/\sigma VV')V = V'P^{-1}P^{-T}V
    cholesky_factor = torch.potrf(shifted_mat)
    # R = P^{-T}V
    R = torch.trtrs(low_rank_mat, cholesky_factor.t(), upper=False)[0]

    return R


def woodbury_solve(vector, woodbury_factor, shift):
    """
    Solves the system of equations:
        (sigma*I + VV')x = b
    Using the Woodbury formula.

    Input:
        - vector (size n) - right hand side vector b to solve with.
        - woodbury_factor (k x n) - The result of calling woodbury_factor on V
          and the shift, \sigma
        - shift (scalar) - shift value sigma
    """
    right = (1 / shift) * woodbury_factor.t().matmul(woodbury_factor.matmul(vector))
    return (1 / shift) * (vector - right)
