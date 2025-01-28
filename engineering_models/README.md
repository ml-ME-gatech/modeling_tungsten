# Overview

Empirical engineering material property models for tungsten alloys are developed in these notebooks using data reported in scientific literature along with standard methods and techniques from regression analysis. **If you would like to begin immediately browsing notebooks please start with [modeling ultimate tensile strength](uts_modeling.ipynb).**

## Summary
The analysis developed in the notebooks consists of (1) exploration/initial visualization (notebooks ending with "exploration") (2) model development (notebooks ending with "modeling") and (3) an example extension \& application (the notebook [parametric_application](parametric_application.ipynb)). The specific material properties are:

1. Ultimate Tensile Stress:the ultimate stress sustained by a material during uniaxial testing, a measure of material strength.
2. Uniform Elongation: the (percent) elongation of the material at the Ultimate Tensile Stress measured during uniaxial testing, a measure of material ductility.
3. Thermal Conductivity: a measure of the material's ability to conduct heat.
4. Creep stress: a measure of the material's endurance to sustained loads over long periods of time.

If you woud like to review the contents of this directory comprehensively, please follow this order:

1. [uts_data_exploration.ipynb](uts_data_expoloration.ipynb)
2. [uts_modeling.ipynb](uts_modeling.ipynb)
3. [ue_data_exploration.ipynb](ue_data_exploration.ipynb)
4. [ue_modeling.ipynb](ue_modeling.ipynb)
5. [conductivity_data_exploration.ipynb](conductivity_data_exploration.ipynb)
6. [conductivity_modeling.ipynb](conductivity_modeling.ipynb)
7. [creep_modeling.ipynb](creep_modeling.ipynb) 

![Parametric representation of hypothetical new material, compared with existing materials. _cf._ [parametric_application.ipynb](parametric_application.ipynb) for details](images/parametric_material_plot.svg)


## Results 

### Ultimate Tensile Stress

![Ultimate Tensile Stress](./.git_images/uts_data_fit.svg)

### Uniform Elongation
![Uniform Elongation](./.git_images/ue_data_fit.svg)

### Conductivity
![Conductivity](./.git_images/conductivity_data_fit.svg)