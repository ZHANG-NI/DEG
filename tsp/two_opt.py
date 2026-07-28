import numpy as np
import numba as nb
import concurrent.futures
from functools import partial
import torch

@nb.njit(nb.float32(nb.float32[:,:], nb.uint16[:], nb.uint16), nogil=True)
def two_opt_once(distmat, tour, fixed_i = 0):
    '''in-place operation'''
    n = tour.shape[0]
    p = q = 0
    delta = 0
    for i in range(1, n - 1) if fixed_i==0 else range(fixed_i, fixed_i + 1):
        for j in range(i + 1, n):
            node_i, node_j = tour[i], tour[j]
            node_prev, node_next = tour[i - 1], tour[(j + 1) % n]
            if node_prev == node_j or node_next == node_i:
                continue
            change = (
                distmat[node_prev, node_j] + distmat[node_i, node_next]
                - distmat[node_prev, node_i] - distmat[node_j, node_next]
            )
            if change < delta:
                p, q, delta = i, j, change
    if delta < -1e-6:
        tour[p: q + 1] = np.flip(tour[p: q + 1])
        return delta
    else:
        return 0.0


@nb.njit(nb.uint16[:](nb.float32[:,:], nb.uint16[:], nb.int64), nogil=True)
def _two_opt_python(distmat, tour, max_iterations=1000):
    iterations = 0
    min_change = -1.0
    while min_change < -1e-6 and iterations < max_iterations:
        min_change = two_opt_once(distmat, tour, 0)
        iterations += 1
    return tour


def batched_two_opt_python(dist: np.ndarray, tours: np.ndarray, max_iterations=1000):
    dist = dist.astype(np.float32)
    tours = tours.astype(np.uint16)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for tour in tours:
            future = executor.submit(partial(_two_opt_python, distmat=dist, max_iterations=max_iterations), tour = tour)
            futures.append(future)
        return np.stack([f.result() for f in futures])


def batched_two_opt_torch(distmat: np.ndarray, tours: np.ndarray, max_iterations=1000, device="cuda:0"):
    """
    Perform batched 2-opt optimization on a set of TSP tours using PyTorch.
    
    Parameters:
    - distmat: np.ndarray (N, N), Precomputed distance matrix.
    - tours: np.ndarray (B, N), Initial batch of TSP tours.
    - max_iterations: int, Maximum number of iterations for 2-opt.
    - device: str, "cpu" or "cuda" for computation.
    
    Returns:
    - np.ndarray (B, N), Optimized batch of TSP tours.
    """
    iterator = 0
    tours = tours.copy()
    
    # Convert to torch tensors
    with torch.inference_mode():
        cuda_distmat = torch.from_numpy(distmat).to(device)  # (N, N)
        cuda_tours = torch.from_numpy(tours).to(device)  # (B, N)
        batch_size, num_points = cuda_tours.shape
        min_change1 = -1.0

        while min_change1 < 0.0 and iterator < max_iterations:
            min_change = 0.0

            # build all candidate (i, j) pairs
            #i_indices, j_indices = torch.triu_indices(num_points - 1, num_points, offset=2, device=device)  # i < j

            # gather the node indices for each (i, j)
            node_i = cuda_tours[:, :-1]  # (B, K)
            node_j = cuda_tours[:, :-1]  # (B, K)
            points_i_plus_1 = cuda_tours[:, 1:]  # (B, K)
            points_j_plus_1 = cuda_tours[:, 1:]  # (B, K)
            
            # compute all candidate deltas
            A_ij = cuda_distmat[node_i.unsqueeze(2).expand(batch_size,-1, num_points-1), node_j.unsqueeze(1).expand(batch_size,num_points-1, -1)]  # New edge (prev -> j)
            A_i_plus_1_j_plus_1 = cuda_distmat[points_i_plus_1.unsqueeze(2).expand(batch_size,-1, num_points-1), points_j_plus_1.unsqueeze(1).expand(batch_size,num_points-1, -1)]  # New edge (i -> next)
            A_i_i_plus_1 = cuda_distmat[node_i, points_i_plus_1].unsqueeze(2).expand(batch_size,-1, num_points-1)  # Old edge (prev -> i)
            A_j_j_plus_1 = cuda_distmat[node_j, points_j_plus_1].unsqueeze(1).expand(batch_size,num_points-1, -1)  # Old edge (j -> next)
            change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
            valid_change = torch.triu(change, diagonal=2)
            #print(A_ij[0] )
            min_change,flatten_argmin_index = torch.min(valid_change.reshape(batch_size, -1), dim=-1)
            #flatten_argmin_index = torch.argmin(valid_change.reshape(batch_size, -1), dim=-1)
            min_i = torch.div(flatten_argmin_index, num_points-1, rounding_mode='floor')
            min_j = torch.remainder(flatten_argmin_index, num_points-1)
            #print(min_change)
            min_change1 = torch.min(valid_change)
            #print(min_i,min_j)
            if min_change1< -1e-6:
                for i in range(batch_size):
                    if min_change[i] < -1e-6:
                        #print(cuda_tours[i])
                        cuda_tours[i, min_i[i] + 1:min_j[i] + 1] = torch.flip(cuda_tours[i, min_i[i] + 1:min_j[i] + 1], dims=(0,))
                        #print(cuda_tours[i])
                iterator += 1
            else:
                break

            if iterator >= max_iterations:
                break

        return cuda_tours.cpu().numpy()  # return the improved tours