## Achieving Byzantine-Resilient Federated Learning via Layer-Adaptive Sparsified Model Aggregation, WACV 2025


## Usage

### Example

Generally, to run a case with default settings, you can easily use the following command:

```
python main.py --attack $attack --defend $defend --dataset $data 
```
Here,

```
attack = {'agrTailoredTrmean', 'agrAgnosticMinMax', 'agrAgnosticMinSum', 'signflip_attack', 'noise_attack', 'random_attack', 'lie_attack', 'byzmean_attack', 'non_attack'}

defend = {'fedavg', 'signguard', 'dnc', 'lasa', 'bulyan', 'tr_mean', 'multi_krum', 'sparsefed', 'geomed'}

data = {'cifar', 'noniidcifar', 'cifar100', 'noniidcifar100'}
```

For example, to run LASA on defending against ByzMean attack on IID CIFAR-10 dataset, you can use:

```
python main.py --attack byzmean_attack --defend lasa --dataset cifar
```

Results will be recorded in `exp_results` folder.

## Hyperparameters

### We list the mainly used hyperparameters as follows.

| Argument        | Type       | Description                                                               |
|-----------------|------------|---------------------------------------------------------------------------|
| `repeat`         | int        | Number of repeat of training                                           |
| `num_attackers`    | int        |  How many clients are malicios, use integer here (e.g., 20 -> 20% of total clients are malicious        |
| `num_users`    | int      | How many clients in the FL system                              |
| `num_selected_users`         | int | how many clients to be selected per round.                                              |
| `round`      | int      | Total training rounds  |
| `tau`      | int      | Local training epochs  |


More detailed hyparameters are presented in paper and you can find them in `config/attack/$data/basee.yaml`.


#### Hyperparameters listed below are specifically for LASA.
| Argument        | Type       | Description                                                               |
|-----------------|------------|---------------------------------------------------------------------------|
| `sparsity`             | store_true        | Pre-aggregation sparsification level                                                      |
| `lambda_n`  | float        | Filtering radius for norm                                                    |
| `lambda_s`  | float        | Filtering radius for sign                                                    |


## Citation
If you find our repository is useful for your work, please cite our work:
```
@article{xu2024achieving,
  title={Achieving Byzantine-Resilient Federated Learning via Layer-Adaptive Sparsified Model Aggregation},
  author={Xu, Jiahao and Zhang, Zikai and Hu, Rui},
  journal={arXiv preprint arXiv:2409.01435},
  year={2024}
}
```
    
## Acknowledgment

We would like to thank the work that help our paper:

1. SignGuard: https://github.com/JianXu95/SignGuard/tree/main.