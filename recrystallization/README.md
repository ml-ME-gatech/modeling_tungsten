# Overview

This project repository contains python scripts and data neccessary to estimate the effect of recrystillization on the macroscopic material properties of tungsten (W) and W alloys. 
The goal is to develop a path function $Y(T(t)): \mathbb{R} \mapsto [0,1]$ which represents the fraction of recrystillized material based upon an arbitary time history $T(t): \mathbb{R}_+ \to \mathbb{R}$. 
Second, a statistical estimate in the reduction of material hardness based on material recrystillization fraction, which is correlated to the material yield strength, an important macroscopic material property is detailed. 

# Summary

## Recrystillization Description 

## Estimating Reduction in Material Hardness
Material yield strength is linearly related to hardness (Tabor's relationship). It's reasonable to expect that a reduction in material hardness will result in a corresponding frational reduction in yield strength. 
Using measured tungsten hardness data during recrystillization experiments, the expected reduction in material hardness is estimated in [hardness_rx_model.ipynb](hardness_rx_model.ipynb)
## Estimating Recrystillization Fraction State Function
### Estimation of Arrhenius Process Parameters via Linear Regression
### Least Squares Comparisons of State Function Models
### Bayesian Calibration of Generalized Logistic Function Model
### Assessing the Extrapolative Accuracy of Calibrated Generalized Logistic Function Model
## Obtaining Path Functions from State Function Models

