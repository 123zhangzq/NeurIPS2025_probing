# Practical Implications: LEHD Regularization Experiment

This folder contains the training code used in the Practical Implications subsection for the LEHD regularization experiment. To reproduce the results, replace the original `TSPModel.py` and `TSPTrainer.py` files in the LEHD codebase with the provided versions.

- In `TSPModel.py`, line 175 applies the L1 regularization loss, computed by the `l1_sparsity_loss` function defined in lines 208–214.
- In `TSPTrainer.py`, line 197 incorporates this loss, with line 196 using `L1_lambda` to control the magnitude of regularization.
