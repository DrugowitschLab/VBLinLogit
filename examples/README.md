VBLinLogit example scripts
==========================

The files starting `vb_linear` provide examples for the use of variational Bayesian linear regression. The files starting in `vb_logit` provide examples for variational Bayesian logistic regression. Calling `vb_examples` results in running all example scripts in this folder.

The scripts `vb_linear_example_highdim`, `vb_linear_example_modelsel`, `vb_linear_example_sparse`, `vb_logit_example_coeff`, `vb_logit_example_highdim` and `vb_logit_example_modelsel` reproduce the figures in *Variational Bayesian
inference for linear and logistic regression*, [arxiv:1310.5438](http://arxiv.org/abs/1310.5438) [stat.ML].

Linear regression examples
--------------------------

* `vb_linear_example`: demonstrates that variational Bayesian linear regression without and with ARD provides more robust fit than least-squares for datasets with uninformative dimensions.

* `vb_linear_example_highdim`: shows how the Bayesian regularization inherit to variational Bayesian linear regression perform beneficial for high-dimensional datasets and few training examples per dimension.

* `vb_linear_example_sparse`: more elaborate version of `vb_linear_example` which demonstrates that ARD is able to detect and ignore uninformative input dimensions, leading to overall better test-set predictions.

* `vb_linear_example_modelsel`: demonstrates how to use the variational bound to comare a set of linear models of varying complexity, and choose the most adequate model for a given dataset.

Logistic regression examples
----------------------------

* `vb_logit_example`: demonstrates that variational Bayesian logistic regression without and with ARD provides a more robust fit than linear discriminant analysis for datasets with uninformative dimensions.

* `vb_logit_example_coeff`: similar to `vb_logit_example`, demonstrates the use variational Bayesian logistic regression to recover the correlations coefficients, and to compare the fits to linear discriminant analysis.

* `vb_logit_example_highdim`: demonstrates that ARD is able to detect and ignore uninformative input dimensions, leading to overall better test-set predictions.

* `vb_logit_example_modelsel`: demonstrates how to use the variational bound to comare a set of logistic models of varying complexity, and choose the most adequate model for a given dataset.