import os
import random
import time

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import scipy
from net1 import Net
from sampler_inference import Sampler, get_subroutes
from utils import load_test_dataset
import scipy.stats as stats
from typing import Tuple, List


EPS = 1e-10

def validate_route(distance: torch.Tensor, demands: torch.Tensor, routes: List[torch.Tensor]) -> Tuple[bool, float]:
    length = 0.0
    valid = True
    visited = {0}
    for r in routes:
        d = demands[r].sum().item()
        if d>1.000001:
            valid = False
        length += distance[r[:-1], r[1:]].sum().item()
        for i in r:
            i = i.item()
            if i<0 or i>=distance.size(0):
                valid = False
            else:
                visited.add(i)  # type: ignore
    if len(visited) != distance.size(0):
        valid = False
    return valid, length


@torch.no_grad()
def infer_instance(model, pyg_data, demands, distances, positions, n_samples, t_iter_diff, k_sparse):
    if model is not None:
        model.eval()
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS
    else:
        heu_mat = None

    sampler = Sampler(
        distances=distances.cpu(),
        demand=demands.cpu(),
        positions=positions.cpu(),
        n_samples=n_samples,
        heuristic=heu_mat.cpu() if heu_mat is not None else heu_mat,
        k_sparse=k_sparse,
        elitist=ACOALG == "ELITIST",
        maxmin=ACOALG == "MAXMIN",
        rank_based=ACOALG == "RANK",
        device="cpu",
        local_search_type="nls",
    )

    results = torch.zeros(size=(len(t_iter_diff),))
    diversities = torch.zeros(size=(len(t_iter_diff),))
    elapsed_time = 0
    for i, t in enumerate(t_iter_diff):
        results[i], diversities[i], t, paths, cost = sampler.run(t)
        path = get_subroutes(sampler.shortest_path)
        valid, length = validate_route(distances, demands, path)
        #assert (length - results[i].item()) < 1e-4  # FIXME: remove this line
        if valid is False:
           print("invalid solution.")
        elapsed_time += t
    return results, diversities, elapsed_time


@torch.no_grad()
def test(dataset, model, n_samples, t_iter, k_sparse):
    _t_iter = [0] + t_iter
    t_iter_diff = [_t_iter[i + 1] - _t_iter[i] for i in range(len(_t_iter) - 1)]

    sum_results = torch.zeros(size=(len(t_iter_diff),))
    sum_diversities = torch.zeros(size=(len(t_iter_diff),))
    sum_times = 0
    i=1
    for pyg_data, demands, distances, positions in tqdm(dataset, dynamic_ncols=True):
        results, diversities, elapsed_time = infer_instance(
            model, pyg_data, demands, distances, positions, n_samples, t_iter_diff, k_sparse
        )
        sum_results += results
        sum_diversities += diversities
        sum_times += elapsed_time
        print(results,sum_results/i)
        i=i+1
    return sum_results / len(dataset), sum_diversities / len(dataset), sum_times / len(dataset)


def main(ckpt_path, n_nodes, k_sparse, size=None, n_samples=100, n_iter=10, guided_exploration=False, seed=0):
    test_list = load_test_dataset(n_nodes, k_sparse, DEVICE, TAM)
    test_list = test_list[:(size or len(test_list))]

    t_iter = list(range(1, n_iter + 1))
    print("problem scale:", n_nodes)
    print("checkpoint:", ckpt_path)
    print("number of instances:", size)
    print("device:", 'cpu' if DEVICE == 'cpu' else DEVICE+"+cpu" )
    print("n_samples:", n_samples)
    print("seed:", seed)

    if ckpt_path is not None:
        net = Net(gfn=True, Z_out_dim=1 if guided_exploration else 1).to(DEVICE)
        net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    else:
        net = None
    avg_cost, avg_diversity, duration = test(test_list, net, n_samples, t_iter, k_sparse)
    print('average inference time: ', duration)
    for i, t in enumerate(t_iter):
        print(f"T={t}, avg. cost {avg_cost[i]}, avg. diversity {avg_diversity[i]}")
    '''
    # Save result in directory that contains model_file
    filename = os.path.splitext(os.path.basename(ckpt_path))[0] if ckpt_path is not None else 'none'
    dirname = os.path.dirname(ckpt_path) if ckpt_path is not None else f'../pretrained/cvrp_nls/{args.nodes}/no_model'
    os.makedirs(dirname, exist_ok=True)
    
    result_filename = f"test_result_ckpt{filename}-cvrp{n_nodes}-ninst{size}-{ACOALG}-nants{n_samples}-niter{n_iter}-seed{seed}"
    result_file = os.path.join(dirname, result_filename + ".txt")
    with open(result_file, "w") as f:
        f.write(f"problem scale: {n_nodes}\n")
        f.write(f"checkpoint: {ckpt_path}\n")
        f.write(f"number of instances: {len(test_list)}\n")
        f.write(f"device: {'cpu' if DEVICE == 'cpu' else DEVICE+'+cpu'}\n")
        f.write(f"n_samples: {n_samples}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"average inference time: {duration}\n")
        for i, t in enumerate(t_iter):
            f.write(f"T={t}, avg. cost {avg_cost[i]}, avg. diversity {avg_diversity[i]}\n")

    results = pd.DataFrame(columns=['T', 'avg_cost', 'avg_diversity'])
    results['T'] = t_iter
    results['avg_cost'] = avg_cost
    results['avg_diversity'] = avg_diversity
    results.to_csv(os.path.join(dirname, result_filename + ".csv"), index=False)
    '''

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", type=int, help="Problem scale")
    parser.add_argument("-k", "--k_sparse", type=int, default=None, help="k_sparse")
    parser.add_argument("-p", "--path", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("-i", "--n_iter", type=int, default=1, help="Number of iterations of Sampler to run")
    parser.add_argument("-n", "--n_samples", type=int, default=100, help="Number of samples")
    parser.add_argument("-d", "--device", type=str,
                        default=("cuda:0" if torch.cuda.is_available() else "cpu"), 
                        help="The device to train NNs")
    parser.add_argument("-s", "--size", type=int, default=None, help="Number of instances to test")
    ### DEG
    parser.add_argument("--disable_guided_exp", action='store_true', help='True for model w/o guided exploration.')
    ### Dataset
    parser.add_argument("--tam", action="store_true", help="Use TAM dataset")
    ### Sampler
    parser.add_argument("--sampler", type=str, default="AS", choices=["AS", "ELITIST", "MAXMIN", "RANK"], help="Sampler algorithm")
    ### Seed
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    if args.k_sparse is None:
        args.k_sparse = args.nodes // (5*10)

    DEVICE = args.device if torch.cuda.is_available() else 'cpu'
    TAM = args.tam
    ACOALG = args.sampler

    # seed everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.path is not None and not os.path.isfile(args.path):
        print(f"Checkpoint file '{args.path}' not found!")
        exit(1)

    main(
        args.path,
        args.nodes,
        args.k_sparse,
        args.size,
        args.n_samples,
        args.n_iter,
        False,#not args.disable_guided_exp,
        args.seed
    )
