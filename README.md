# RS-Reg: Probabilistic and Robust Certified Regression Against Adversarial Attack Through Randomized Smoothing

## Overview
This GitHub repository was established to present the results of the very first work on the certification of regression models against adversarial attack (L2-Attack) using randomized smoothing. While randomized smoothing has been greatly developed for classification tasks, its potential for regression tasks has not been explored. In this work, we show how the maximum adversarial perturbation of the input data can be derived for the base regression model, and then we show how a similar certificate can be extracted for the averaging function as a basic smoothing. 

This repository contains a Jupyter Notebook file for figures presented in the paper for a synthetic mapping function.  It also contains a Python script for the certification of camera-relocation task using RGB images using the code and model provided in [DSAC*](https://github.com/vislearn/dsacstar) repository.

We included all the files/models/datasets required to run the project and there is no need to download external files. 

## Installation
Synthetic simulation requires the following Python libraries/packages:
```
plotly (5.18.0)
matplotlib (3.8.2)
numpy (1.26.2)
scipy (1.11.4)
```
To run the Camera re-localization pipeline, all the packages suggested in  [DSAC*](https://github.com/vislearn/dsacstar) as well as the following packages repository should be installed:
```
scipy (1.11.4)
matplotlib (3.8.2)
```

## Project Structure

The repository is organized as follows:

- **`notebooks/`:** Jupyter notebook for evaluating synthetic regression function.
- **`dsacstar/test_certificate.py`:** Python script for evaluating the robustness of camera re-localization model.
- **`dsacstar/datasets/`:** Contains the dataset used for evaluation. Download it [here](https://github.com/vislearn/dsacstar).
- **`dsacstar/newmodels/`:** Contains pre-trained models for the Cambridge landmarks dataset. Download it [here](https://heidata.uni-heidelberg.de/file.xhtml?persistentId=doi:10.11588/data/N07HKC/CBK0OL).

For extracting figures of synthetic function, please go to the `notebook/` directory, open `synthetic.ipynb` file, and run all the cells.

For running the camera re-localization code, please go to the `dsactstar\` directory in Anaconda Prompt and run:
 ```
python test_certificate.py Cambridge_GreatCourt newmodels\rgb\cambridge\Cambridge_GreatCourt.net
```
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
