# Diffusion-Enhanced GFlowNet for Solving Vehicle Routing Problems

This paper has been accepted by **NeurIPS 2026**: [OpenReview](https://openreview.net/forum?id=5pGVh6PbGd)

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

If you find our work helpful, please consider citing the following paper:

```bibtex
@inproceedings{zhang2026deg,
  title     = {Diffusion-Enhanced GFlowNet for Solving Vehicle Routing Problems},
  author    = {Zhang, Ni and Zhang, Zhiqin and Pan, Ling and Lau, Hoong Chuin and Cao, Zhiguang},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=5pGVh6PbGd}
}
```
