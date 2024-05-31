# MMD-FUSE

This package implements the MMD-FUSE test for two-sample testing, as proposed in our paper MMD-FUSE: Learning and Combining Kernels for Two-Sample Testing Without Data Splitting.
The experiments of the paper can be reproduced using the [mmdfuse-paper](https://github.com/antoninschrab/mmdfuse-paper/) repository.

The implementation of MMD-FUSE is in Jax, which runs 500 times faster using a GPU (results from the notebook [demo_speed.ipynb](https://github.com/antoninschrab/mmdfuse-paper/blob/master/demo_speed.ipynb) in the [mmdfuse-paper](https://github.com/antoninschrab/mmdfuse-paper/) repository).

The notebook also contains a demo showing how to use our MMDAgg test.
We also provide installation instructions and example code below.

| Speed in s | Jax (GPU) | Jax (CPU) |
| -- | -- | -- |
| MMD-FUSE | 0.0054 | 2.95 |

## Requirements

The requirements for the Jax version are:
- `python 3.9`
  - `jax`
  - `jaxlib`

## Installation

First, we recommend creating a conda environment:
```bash
conda create --name mmdfuse-env python=3.9
conda activate mmdfuse-env
# can be deactivated by running:
# conda deactivate
```

We then install the required depedencies ([Jax installation instructions](https://github.com/google/jax#installation)) by running either:
- for GPU:
  ```bash
  pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  # conda install -c conda-forge -c nvidia pip numpy scipy cuda-nvcc "jaxlib=0.4.1=*cuda*" jax
  ```
- or, for CPU:
  ```bash
  conda install -c conda-forge -c nvidia pip numpy scipy cuda-nvcc jaxlib=0.4.1 jax
  ```
  
Our `mmdfuse` package can then be installed as follows:
```bash
pip install git+https://github.com/antoninschrab/mmdfuse.git
```

## MMD-FUSE

**Two-sample testing:** Given arrays X of shape $(N_X, d)$ and Y of shape $(N_Y, d)$, our MMDAgg test `mmdfuse(X, Y, key)` returns 0 if the samples X and Y are believed to come from the same distribution, and 1 otherwise.

**Jax compilation:** The first time the function is evaluated, Jax compiles it. 
After compilation, it can fastly be evaluated at any other X and Y of the same shape. 
If the function is given arrays with new shapes, the function is compiled again.
For details, check out the [demo_speed.ipynb](https://github.com/antoninschrab/mmdfuse-paper/blob/master/demo_speed.ipynb) notebook on the [mmdagg-paper](https://github.com/antoninschrab/mmdfuse-paper/) repository.

```python
# import modules
>>> import jax.numpy as jnp
>>> from jax import random
>>> from mmdfuse import mmdfuse

# generate data for two-sample test
>>> key = random.PRNGKey(0)
>>> key, subkey = random.split(key)
>>> subkeys = random.split(subkey, num=2)
>>> X = random.uniform(subkeys[0], shape=(500, 10))
>>> Y = random.uniform(subkeys[1], shape=(500, 10)) + 1

# run MMD-FUSE test
>>> key, subkey = random.split(key)
>>> output = mmdfuse(X, Y, subkey)
>>> output
Array(1, dtype=int32)
>>> output.item()
1
>>> output, p_value = mmdfuse(X, Y, return_p_val=True)
>>> output
Array(1, dtype=int32)
>>> p_value
Array(0.00049975, dtype=float32)
```

## Contact

If you have any issues running our MMD-FUSE test, please do not hesitate to contact [Antonin Schrab](https://antoninschrab.github.io).

## Affiliations

Centre for Artificial Intelligence, Department of Computer Science, University College London

Gatsby Computational Neuroscience Unit, University College London

Inria London

## Bibtex

```
@article{biggs2023mmdfuse,
  author        = {Biggs, Felix and Schrab, Antonin and Gretton, Arthur},
  title         = {{MMD-FUSE}: {L}earning and Combining Kernels for Two-Sample Testing Without Data Splitting},
  year          = {2023},
  journal       = {Advances in Neural Information Processing Systems},
  volume        = {36}
}
```

## License

MIT License (see [LICENSE.md](LICENSE.md)).

## Related tests

- [mmdagg](https://github.com/antoninschrab/mmdagg/): MMD Aggregated MMDAgg test
- [ksdagg](https://github.com/antoninschrab/ksdagg/): KSD Aggregated KSDAgg test
- [agginc](https://github.com/antoninschrab/agginc/): Efficient MMDAggInc HSICAggInc KSDAggInc tests
- [dpkernel](https://github.com/antoninschrab/dpkernel/): Differentially private dpMMD dpHSIC tests
- [dckernel](https://github.com/antoninschrab/dpkernel/): Differentially private dpMMD dpHSIC tests
