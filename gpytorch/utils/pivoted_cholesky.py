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
