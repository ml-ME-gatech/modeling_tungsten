# Overview

This project repository contains python scripts and data neccessary to estimate the effect of recrystillization on the macroscopic material properties of tungsten (W) and W alloys. 
The goal is to develop a path function $Y(T(t)): \mathbb{R} \mapsto [0,1]$ which represents the fraction of recrystillized material based upon an arbitary time history $T(t): \mathbb{R}_+ \to \mathbb{R}$. 
Second, a statistical estimate in the reduction of material hardness based on material recrystillization fraction, which is correlated to the material yield strength, an important macroscopic material property is detailed. 

# Summary

## Recrystillization Description 
Recrystallization of the warm-rolled tungsten plate is a thermally activated _phase-change_ process governed by jumps of individual tungsten atoms. At a macroscopic level, the recrystillization fraction $X$ in an _isothermal_ experiment is measured by monitoring the material hardness over time, judging the phase using the law of mixtures. The simplest mathematical model for this process begins with a spatial poisson distributed nucleation sites of the recrystillized material that grow exponentially until the entire volume of the new material is consumed by this new phase, see below figure.

![Simplified grain growth from nucleation sites model](images/grain_growth.gif)

## Estimating Reduction in Material Hardness
Material yield strength is linearly related to hardness (Tabor's relationship). It's reasonable to expect that a reduction in material hardness will result in a corresponding frational reduction in yield strength. 
Using measured tungsten hardness data during recrystillization experiments, the expected reduction in material hardness is estimated in [hardness_rx_model.ipynb](hardness_rx_model.ipynb) and assumed to be less than $\mathbf{22}$ \%.
## Estimating Recrystillization Fraction State Function
We assess the fitting and extrapolative capabilities of two models (1) The (modified) Johnson–Mehl–Avrami–Kolmogorov (JMAK) model and (2) a generalized logistic model to fit observed experimental data using least squares regression techinques in [initial_least_squares_comparison.ipynb](initial_least_squares_comparison.ipynb). The model forms are introduced, and their capabilities discussed in these notebooks. I found the capabilities to be fairly simimlar, however the generalized logistic model is bijective and continuous, which is convinient for future applications, so I favored

### Estimation of Arrhenius Process Parameters via Linear Regression
Both models are fairly nonlinear, and have products and transforms of exponential dependencies on time and temperature. I found it useful to obtain good initial guesses for some parameters by linearizing the model, or approximating parameters from data in the notebooks [arrhenius_process_esimation.ipynb](arrhenius_process_estimation.ipynb).

### Bayesian Calibration of Generalized Logistic Function Model
TO - DO
### Assessing Extrapolative Predictive Inference
TO - DO
### Extrapolation to Tungsten Alloys with Limited Data
TO - DO
## Obtaining Path Functions from State Function Models
TO - DO

