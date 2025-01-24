# Overview
This markdown summarizes some applications where the results from the calibration and model development prove useful.

TO-DO:
1. Calibration to alloys with global model
2. Demonstrate effect of latent variables on recrystillization temperature

### 1. Time-to-Recrystillization Estimation
Using the calibrated models we can **predict the _time to recrytillization fraction_ for any $X = X^*$**. A sample of this sort of analysis is shown in the figure below, using the JMAK models calibrated independently to each data set, for a recrystillization fraction $X^* = 0.9$. The interpretation being that at the temperature specified on the horizontal axis, it will take the amount of time specified on the y-axis to reach a recrystillization fraction of $X^* = 0.9$. Note the different temperature ranges on the horizontal axis, clearly the two tungsten's on the left panel are more resistant to recrystilization. The gray envelopes are 95% confidence intervals.

![Time to Recrystillization](.git_images/jmak_ttr.svg)

_Time to recrystillization of_ $X^* = 0.9$ _for different tungstens as a function of temperature_

If we fix our time at $t_{operating} = 1$ year, this corresponds to a unique temperature for each alloy. These temperatures are summarized in the below tables for the JMAK and GL models with 95% confidence intervals shown: ML means "most likely", an estimate made using the maximum likelihood estimate of the model paramters. The greater uncertainty in Richou et al.'s data means that the confidence intervals are considerably larger. Surprisingly, using the GL model produces much greater temperature estimates than the JMAK model for Lopez et al. (2015) - HR. 

**JMAK 1-year Recrystillization Temperature** [$^\circ$ C]: The temperature required to achieve a recrystillization fraction of $0.9$ after $1$ year.
|                                |   ML |   Lower 95\% |   Upper 95% |
|:-------------------------------|-----:|-------------:|------------:|
| Richou et al. (2020) - Batch A | 1117 |         1081 |        1149 |
| Lopez et al. (2015) - MR       | 1113 |         1108 |        1114 |
| Richou et al. (2020) - Batch B | 1076 |         1017 |        1120 |
| Lopez et al. (2015) - HR       |  937 |          933 |         952 |
| Yu et al. (2017)               |  N/A |          N/A |         914 |


**Generalized Logistic 1-year Recrystillization Temperature** [$^\circ C$]: The temperature required to achieve a recrystillization fraction of $0.9$ after $1$ year.
|                                |   ML |   Lower 95\% |   Upper 95% |
|:-------------------------------|-----:|-------------:|------------:|
| Richou et al. (2020) - Batch A | 1123 |         1082 |        1145 |
| Lopez et al. (2015) - MR       | 1115 |         1109 |        1115 |
| Richou et al. (2020) - Batch B | 1075 |         1016 |        1107 |
| Lopez et al. (2015) - HR       |  968 |          944 |         982 |
| Yu et al. (2017)               |  867 |          N/A |         853 |


### 2. Non-isothermal Model Analysis

#### 2a. Modeling Recrystillization fraction $Y(t)$ using Arbitary temperature histories $T(t)$
It is often the case that the temperature experienced by the tungsten is not isothermal in real applications. We can use the model developed and validated in [nonisothermal_modeling](/nonisothermal_modeling) to **predict the recrystillization fraction based upon an arbitary temperature history**. This is demonstrated in the below figure. The first (upper left panel) compares the nonisothermal model predictions for the isothermal case with the experimental data comparing both the JMAK and GL models. The recrystillization in the linear decreasing temperature case proceeds much more quickly at the beggining due to the higher temperature, while slowing towards the end, with the opposite trend experienced in the linear increasing. The exponential is similar in nature to the linear decreasing temperature.

![Non-isothermal Fraction Prediction](.git_images/jmak_glm_comparison.svg)

_Nonisothermal recrystilization fractions using several different temperature profiles_

#### 2b. Non-isothermal Model Interpretation

### 3. Using the Combined Model 

#### 3a. Extending Combined Model to Calibrate with Limited Data
#### 3b. Extrapolating to Future Tungsten/Tungsten Alloys