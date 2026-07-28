# DEG: Diffusion-Enhanced GFlowNet for Vehicle Routing Problems

## Training (DEG)
```bash
TSP : cd tsp  && python train1.py <n_nodes>     
CVRP: cd cvrp && python train1.py <n_nodes>
```

## Inference / reproducing tables
```bash
TSP : cd tsp  && python test.py  <n_nodes> -p <ckpt> -n 100 -i 10 --seed 1
CVRP: cd cvrp && python test1.py <n_nodes> -p <ckpt> -n 100 -i 10 --seed 1
```
