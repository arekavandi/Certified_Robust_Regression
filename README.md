[![arXiv](https://img.shields.io/badge/arXiv-2405.08892-b31b1b.svg)](https://arxiv.org/abs/2405.08892)

# RS-Reg: Probabilistic and Robust Certified Regression Through Randomized Smoothing
## (Accepted in Transactions on Machine Learning Research)
## Overview
This GitHub repository was established to present the results of the very first work on the certification of regression models against adversarial attack (L2-Attack) using randomized smoothing. While randomized smoothing has been greatly developed for classification tasks, its potential for regression tasks has not been explored. In this work, we show how the maximum adversarial perturbation of the input data can be derived for the base regression model, and then we show how a similar certificate can be extracted for the averaging function as a basic smoothing. 

This repository contains a Jupyter Notebook file for figures presented in the paper for a synthetic mapping function.  It also contains a Python script for the certification of camera-relocation task using RGB images using the code and model provided in [DSAC*](https://github.com/vislearn/dsacstar) repository.

We included all the files/models/datasets required to run the project and there is no need to download external files. 

## Installation
Synthetic simulation requires the following Python libraries/packages:
```
plotly
matplotlib
numpy 
scipy 
```
To run the Camera re-localization pipeline, all the packages suggested in  [DSAC*](https://github.com/vislearn/dsacstar) as well as the following packages repository should be installed:
```
scipy (1.7.3)
matplotlib (3.5.0) 
```
The repository contains an `environment.yml` for use with Conda. Perform the following steps:

1. Clone the repository:
```
git clone https://github.com/arekavandi/Certified_Robust_Regression.git
```
2. Go to the project folder and make a new environment and install the required packages by
```
cd Certified_Robust_Regression
conda env create -f environment.yml
```
3. Change the environment
```
conda activate cerroreg
```
4. To be able to work with DSAC*, you have to install a custom C++ extension. To do so, you must run
```
cd dsacstar
cd dsacstar
python setup.py install
```
Check [DSAC*](https://github.com/vislearn/dsacstar) repository if you get an error!

5. Go back to the directory where the certification `.py` function exists.

Now you are ready to perform the certification!
## Project Structure

The repository is organized as follows:

- **`notebooks/`:** Jupyter notebook for evaluating synthetic regression function.
- **`dsacstar/test_certificate.py`:** Python script for evaluating the robustness of camera re-localization model.
- **`dsacstar/datasets/`:** Contains the dataset used for evaluation. Download it [here](https://drive.google.com/drive/folders/1gmG9rt5aMVg3q7bw8znn199JmglB9x5I?usp=sharing).
- **`dsacstar/newmodels/`:** Contains pre-trained models for the Cambridge landmarks dataset. Download it [here](https://heidata.uni-heidelberg.de/file.xhtml?persistentId=doi:10.11588/data/N07HKC/CBK0OL).
- **`dsacstar/dsacstar/`:** Contains custom C++ extension to run DSAC* visual positioning model.

For extracting figures of synthetic function, please go to the `notebook/` directory, open `synthetic.ipynb` file, and run all the cells.

For running the camera re-localization code, please go to the `dsactstar/` directory in Anaconda Prompt and run:
 ```
python test_certificate.py Cambridge_GreatCourt newmodels\rgb\cambridge\Cambridge_GreatCourt.net
```
## Main Idea
We aim to find the upper bound on the input perturbation (w.r.t. a norm) such that the output values of a regression model stay  with probability P within the accepted region defined by the user (see the below Figure).

![image](https://github.com/arekavandi/Certified_Robust_Regression/assets/101369948/78d5cfce-5ba4-4343-924c-e2253fcaef20)


## Experimental Results
we considered a mapping function $f:\mathbb{R}^2\rightarrow \mathbb{R}$ given by $ f(\textbf{x}=[x_1,x_2]^\top)=10\sin(2x_1)+(x_2-2)^2+15$. This function was investigated for the interval $(-1, 5)$ and the following Figure illustrates the derived bounds for the base regression model (blue), smoothed regression model (red), and discounted smoothed model (black and green) in integer points of the defined domain.

![image](https://github.com/arekavandi/Certified_Robust_Regression/assets/101369948/ed6ba9fe-036f-43a6-87b3-d9788442f19b)


For more experiments on a real application (camera pose estimation), download the paper and read the experimental results section as well as Appendix E. 

# Citations
If you found this GitHub page helpful, please cite the following papers:
```
@article{rekavandi2025rs,
  title={RS-Reg: Probabilistic and Robust Certified Regression Through Randomized Smoothing},
  author={Rekavandi, Aref Miri and Ohrimenko, Olga and Rubinstein, Benjamin IP},
  journal={TMLR},
  year={2025}
}
```
