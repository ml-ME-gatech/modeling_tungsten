# Overview
This project is concerned with modeling material structural and thermal properties for tungsten and tungsten alloys. In particular the material ultimate tensile stress (UTS), $S_u$ the material uniform elongation $\varepsilon_u$, the material creep stress $S_t$, and the conductivity $k$. Empirical models are proposed for incorporating the effect of material recrystallization, a phase change process that depends upon time and temperature. The primary statistical methods used in the repository are techniques from linear regression (model selection, residual analysis, model adquecy checking) and bayesian statistics (MCMC sampling, hierarchical modeling). 

## Highlights

### Material Recrystallization



#### Isothermal Recrystallization


The figure below shows predicted isothermal recrystallization phase fraction $X: (t,T) \to [0,1]$ as a function of time at two different isotherms, with the model calibrated to data from [A. Lopez (2015)](https://orbit.dtu.dk/en/publications/thermal-stability-of-warm-rolled-tungsten). The solid black lines shows the maximum likelihood predictions while the dotted black lines show 95\% confidence intervals based on sampling from the posterior predictive distribution. 

![Example Phase Fraction](./recrystallization/.git_images/Generalized%20Logistic_Lopez%20et%20al.%20(2015)%20-%20MR_data_example.svg)

The parameter $\overline{k}$ developed [here](./recrystallization/model_inference/latent_variable_selection.ipynb) describes how quickly the rate of recrystalliation proceeds, smaller $\overline{k}$ means that recrystallization proceeds more quickly, while larger $\overline{k}$ means it is slower (which is desirable for engineering applications). The below figure compares posterior distributions of this parameter for tungstens processed using different routes obtained via sampling from a bayesian hierarcichal model:

![Recrystallization Rates](/recrystallization/.git_images/latent_variable_posterior_extension.svg)

### State Properties

#### Parametric Modeling
![Parametric representation of hypothetical new material, compared with existing materials. _cf._ [parametric_application.ipynb](parametric_application.ipynb) for details](./.git_images/parametric_representation.svg)

### Coupling Recystallization
The below plots demonstrate the modeled effect of material recrystallization (with $X= 1$) on material property state functions $S_u$ and $\varepsilon_u$.

![As recieved and fully recystallized ultimate tensile stress and uniform elongation](/engineering_models/.git_images/K-W3pRe_Plate_(H)_UTS_and_UE.svg)

## Summary
The listed material properties are dependent on their temperature. Further, the structural properties, $S_u$, $\varepsilon_u$ and $S_t$ are functions of recrystillization phase fraction $X$ which describes the fraction of material that is _recrystallized_. Material _recrystallization_ describes a solid state phase change of material and is a temperature and _time_ dependent process. Material recrystallization is characterized using the material phase fraction $X: (t,T) \to [0,1]$ where $t$ is the time and $T$ is the isothermal material temperature.

The aim of this directory is to obtain reasonable estimates for:

1. A material conductivity state function $k(T)$
2. A creep stress state function $S_t(T)$
3. An ultimate tensile stress path function $S_u(T(t))$
4. A uniform elongation path function $\varepsilon_u(T(t))$

where $T$ is the temperature state of the material and $T(t)$ is an arbitary temperature history. A reasonable model for $S_u(T(t))$ is 

$
S_u(T(t)) = S_{u,0}(T(t)) - Y(T(t)) \Delta_{rx}S_{u}
$ 

Where $\Delta_{rx} S_u$ is the constant decrement in $S_u$ due to full recrystallization at $X= 1$, and $Y(T(t))$ is the _nonisothermal_ recrystallization, with a similar relationship for $\varepsilon_{u}(T(t))$. Models for $S_{u,0}(T)$, $Y(T(t))$ and $\Delta_{rx} S_u$ may be estimated from seperate experiments.

### State Property Estimation
The state properties $k(T),S_t(T),S_{u,0}(T),\varepsilon_{u,0}(T)$ are modeled empirically using data reported in scientific literature along with standard methods and techniques from linear regression analysis in the folder [engineering_models](./engineering_models/). 

The isothermal recrystallization fraction $X(t,T)$ is highly nonlinear in both arguments, and the paramters for the model of $X(t,T)$ are estimated using bayesian statistics and MCMC methods, detailed in the notebooks in [recrystallization](./recrystallization/).

Both of these subfolders have `README.md`'s with more detailed information regarding the model estimation procedures, and description of the results.

### Path Function Estimation
Leveraging the principle of additivity, a simple extension to the isothermal recrystallization fraction $X(t,T)$ leads to the nonisothermal recrystallization fraction $Y(T(t))$ in continuous closed form, developed [here](./recrystallization/nonisothermal_modeling/).

### Parametric Characterization
Different tungsten alloys, and even tungsten processed in two different manners, exhibit different temperature dependent property models. When not at fixed temperature, this makes it difficult to judge how much "better" one alloy is than another, especially when there are multiple material properties that might impact the designed structure. I propose an **empirical parametric modeling approach**  [here](./engineering_models/visualizations/parametric_application.ipynb) to interpolate between different alloy state properties. I also propose a **Bayesian hierarchical model** that allows interpolation in recrystallization  response between observed experiments [here](./recrystallization/model_inference/hierarchical_recystillization_inference.ipynb) using two scalar variables.