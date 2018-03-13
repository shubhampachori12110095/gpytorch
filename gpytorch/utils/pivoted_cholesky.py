import torch


def pivoted_cholesky(matrix, max_iter, error_tol=1e-5):
    matrix_size = matrix.size(-1)
    matrix_diag = matrix.diag()
    error = torch.norm(matrix_diag, 1)
    permutation = torch.arange(len(matrix)).long()

    m = 0
    L = torch.zeros(max_iter, matrix_size)
    while m < max_iter and error > error_tol:
        max_diag_value, max_diag_index = torch.max(matrix_diag[permutation][m:], 0)
        max_diag_index = max_diag_index + m

        pi_m = permutation[m]
        permutation[m] = permutation[max_diag_index][0]
        permutation[max_diag_index] = pi_m

        pi_m = permutation[m]

        L[m, permutation[m]] = torch.sqrt(max_diag_value)[0]
        row = matrix[permutation[m]]
        for i in range(m + 1, matrix_size):
            # TODO: BATCH THIS FOR LOOP!!!
            pi_i = permutation[i]
            L[m, pi_i] = row[pi_i]
            if m > 0:
                L[m, pi_i] -= (torch.sum(L[:m, pi_m] * L[:m, pi_i]))
            L[m, pi_i] /= L[m, pi_m]
            matrix_diag[pi_i] = matrix_diag[pi_i] - (L[m, pi_i] ** 2)

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
