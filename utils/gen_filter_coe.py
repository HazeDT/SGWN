import torch
import numpy as np
from torch_geometric.utils import get_laplacian
from scipy import sparse
from scipy.sparse.linalg import lobpcg

def gen_filter_coe(dataset, num_nodes, s,  n, r, device):
    '''
    :param dataset: pygeometric dataset
    :return: filter coefficent
    '''
    Q = 2
    dataset = dataset[0]
    L = get_laplacian(dataset.edge_index, num_nodes=num_nodes, normalization='sym')
    L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))

    lobpcg_init = np.random.rand(num_nodes, 1)
    lambda_max, _ = lobpcg(L, lobpcg_init)
    lambda_max = lambda_max[0]
    lambda_min = lambda_max / 2

    scale_J = 2 / lambda_min
    scale_1 = 1 / lambda_max

    scale_j = lambda x: scale_1 * np.power(scale_J / scale_1, (x - 1) / (s - 1))
    J = [i + 1 for i in range(s)]  # scale list

    # get matrix operators
    d = get_operator(L, n, s, scale_j, J, Q, lambda_max)

    # enhance sparseness of the matrix operators (optional)
    # d[np.abs(d) < 0.001] = 0.0
    # store the matrix operators (torch sparse format) into a list: row-by-row
    d_list = list()

    for i in range(r):
        d_list.append(scipy_to_torch_sparse(d[i, 0]).to(device))

    return d_list




@torch.no_grad()
def scipy_to_torch_sparse(A):
    A = sparse.coo_matrix(A)
    row = torch.tensor(A.row)
    col = torch.tensor(A.col)
    index = torch.stack((row, col), dim=0)
    value = torch.Tensor(A.data)
    return torch.sparse_coo_tensor(index, value, A.shape)


# function for pre-processing
def ChebyshevApprox_KF(f, n,scale,lambda_max):  # assuming f : [0, pi] -> R
    quad_points = 500
    c = np.zeros(n)
    for k in range(1, n + 1):
        Integrand = lambda x: np.cos((k - 1) * x) * f(scale * lambda_max * (np.cos(x) + 1))
        x = np.linspace(0, np.pi, quad_points)
        y = Integrand(x)
        c[k - 1] = 2 / np.pi * np.trapz(y, x)
    return c

def ChebyshevApprox_SF(f, n,lambda_max):  # assuming f : [0, pi] -> R
    quad_points = 500
    c = np.zeros(n)

    for k in range(1, n + 1):
        Integrand = lambda x: np.cos((k - 1) * x) * f(lambda_max * (np.cos(x) + 1))
        x = np.linspace(0, np.pi, quad_points)
        y = Integrand(x)
        c[k - 1] = 2 / np.pi * np.trapz(y, x)
    return c


# function for pre-processing
def get_operator(L, n, s, scale_j, J, Q, lambda_max):
    r = s+1
    # Mexhat wavelet kernel function
    KF = lambda x: (x*lambda_max) * np.exp(-x*lambda_max)
    SF = lambda x: np.exp(-1) * np.exp(-np.power((x*Q) / (0.6 * lambda_max), 4)) # define the scale function


    c = [None] * r
    c[0] = ChebyshevApprox_SF(SF, n, lambda_max)
    for j in range(1,r):
        scale = scale_j(J[j-1])
        c[j] = ChebyshevApprox_KF(KF, n, scale, lambda_max)

    FD1 = sparse.identity(L.shape[0])
    d = dict()
    Lev = 1

    # for l in range(1, Lev + 1):

    for l in range(1, Lev + 1):

        for j in range(r):
            if j == 0:
                scale = 0
            else:
                scale = scale_j(J[j-1])

            T0F = FD1
            T1F = ((2 * (2*scale/lambda_max - 1)) * L) @ T0F - T0F
            d[j, l - 1] = (1 / 2) * c[j][0] * T0F + c[j][1] * T1F
            for k in range(2, n):
                TkF = ((2 * (2 * scale / lambda_max - 1)) * L) @ T1F - T0F
                T0F = T1F
                T1F = TkF
                d[j, l - 1] += c[j][k] * TkF
        FD1 = d[0, l - 1]

    return d