# Overview
This project is concerned with modeling material structural and thermal properties for tungsten and tungsten alloys. In particular the material ultimate tensile stress (UTS), $S_u$ the material uniform elongation $\varepsilon_u$, the material creep stress $S_t$, and the conductivity $k$.

## Summary
The listed material properties are dependent on their temperature. Further, the structural properties, $S_u$, $\varepsilon_u$ and $S_t$ are functions of recrystillization phase fraction $X$ which describes the fraction of material that is _recrystallized_. Material _recrystallization_ describes a solid state phase change of material and is a temperature and _time_ dependent process.

The aim of this directory is to obtain reasonable estimates for:

1. A material conductivity state function $k(T)$
2. A creep stress state function $S_t(T)$
3. An ultimate tensile stress path function $S_u(T(t))$
4. A uniform elongation path function $\varepsilon_u(T(t))$

where $T$ is the temperature state of the material and $T(t)$ is an arbitary temperature history. 

### Material Recrystallization

#### Isothermal Recrystallization
![Example Phase Fraction](./recrystallization/.git_images/Generalized%20Logistic_Lopez%20et%20al.%20(2015)%20-%20MR_data_example.svg)

### Non-isothermal Recrystallization
![Non-isothermal Fraction Prediction](/recrystallization/.git_images/jmak_glm_comparison.svg)

### State Properties

#### Parametric Modeling
![Parametric representation of hypothetical new material, compared with existing materials. _cf._ [parametric_application.ipynb](parametric_application.ipynb) for details](./engineering_models/images/parametric_material_plot.svg)

#### Coupling Recystallization
![As recieved and fully recystallized ultimate tensile stress and uniform elongation](/engineering_models/.git_images/K-W3pRe_Plate_(H)_UTS_and_UE.svg)

