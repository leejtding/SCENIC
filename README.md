# Self-Consistent Equation-guided Neural Networks for Interval Censored Time-to-Event Data

The code includes the experiments of the Self-Consistent Equation-guided Neural Networks(SCENE) by Sehwan Kim, Rui Wang and Wenbin Lu. We propose a novel deep learning approach to non-parametric estimation of the conditional survival functions using the generative adversarial networks leveraging self-consistent equations. The proposed method is model-free and does not require any parametric assumptions on the structure of the conditional survival function.

## Description: PO Model, High-Dimensional, High-Censoring Case

The `simulations.py` script replicates the results for the high-dimensional case presented in the paper. To run the simulations for each scenario, execute the following command:

```bash
for seed_value in range(1,number of simulations):
    ! python simulations.py --seed $seed_value --ntrain 4000 --C 5 --dim 100 --TI 20 --Model "PO" --VS "T"
```

where ntrain: Dataset size, C: Censoring rate, dim: Number of covariate dimensions, TI: Number of epochs after variable selection, Model: Chooses between the PH and PO models.
