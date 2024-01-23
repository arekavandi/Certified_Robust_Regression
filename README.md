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
- **`dsacstar/datasets/`:** Contains the dataset used for evaluation. Download it [here](https://drive.google.com/drive/folders/1gmG9rt5aMVg3q7bw8znn199JmglB9x5I?usp=sharing).
- **`dsacstar/newmodels/`:** Contains pre-trained models for the Cambridge landmarks dataset. Download it [here](https://heidata.uni-heidelberg.de/file.xhtml?persistentId=doi:10.11588/data/N07HKC/CBK0OL).

For extracting figures of synthetic function, please go to the `notebook/` directory, open `synthetic.ipynb` file, and run all the cells.

For running the camera re-localization code, please go to the `dsactstar/` directory in Anaconda Prompt and run:
 ```
python test_certificate.py Cambridge_GreatCourt newmodels\rgb\cambridge\Cambridge_GreatCourt.net
```
## Main Idea
We aim to find the upper bound on the input perturbation (w.r.t. a norm) such that the output values of a regression model stay  with probability P within the accepted region defined by the user (see the below Figure).

![image](https://github.com/arekavandi/Certified_Robust_Regression/assets/101369948/4e07fc0e-32d7-48b4-8211-64dff03b7848)

## Experimental Results
we considered a mapping function $f:\mathbb{R}^2\rightarrow \mathbb{R}$ given by $ f(\textbf{x}=[x_1,x_2]^\top)=10\sin(2x_1)+(x_2-2)^2+15$. This function was investigated for the interval $(-1, 5)$ and the following Figure illustrates the derived bounds for the base regression model (blue), smoothed regression model (red), and discounted smoothed model (black and green) in integer points of the defined domain.

![image](https://github.com/arekavandi/Certified_Robust_Regression/assets/101369948/2472d3f7-905f-49e0-90db-0da2d864ce1e)

For more experiments on a real application (camera pose estimation), download the paper and read the experimental results section as well as Appendix E. 

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
