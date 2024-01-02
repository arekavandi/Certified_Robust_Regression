# RS-Reg: Probabilistic and Robust Certified Regression Against Adversarial Attack Through Randomized Smoothing

## Overview
This GitHub repository was established to present the results of the very first work on the certification of regression models using randomized smoothing. While randomized smoothing has been greatly developed for classification tasks, its potential for regression tasks has not been explored. In this work, we show how the maximum adversarial perturbation of the input data can be derived for the base regression model, and then we show how a similar certificate can be extracted for the averaging function as a basic smoothing. 

This repository contains a Jupyter Notebook file for figures presented in the paper for a synthetic mapping function.  It also contains a Python script for the certification of camera-relocation task using RGB images using the code provided for [DSAC*](https://github.com/vislearn/dsacstar) repository.

We included all the files/models/datasets required to run the project and there is no need to download external files. 

## Installation

## Project Structure

The repository is organized as follows:

- **`notebooks/`:** Jupyter notebook for evaluating synthetic regression function.
- **`dsacstar/test_certificate.py`:** Python script for evaluating the robustness of camera-relocalization model.
- **`dsacstar/datasets/`:** Contains the dataset used for evaluation.
- **`dsacstar/newmodels/`:** Contains pre-trained models for the Cambridge landmarks dataset.

## Experimental Results

# Citations
If you found this page helpful, please cite the following survey papers:
```
@article{rekavandi2024rsreg,
  title={RS-Reg: Probabilistic and Robust Certified Regression Against Adversarial Attack Through Randomized Smoothing},
  author={Rekavandi Miri, Aref, et al.},
  journal={arXiv preprint arXiv:???},
  year={2024}
}
```
