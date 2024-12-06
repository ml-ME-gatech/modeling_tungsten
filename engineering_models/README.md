## Summary

The notebooks in this directory are concerned with developing empirical engineering material property models for tungsten alloys using data available form the literature. The analyis of the consists of (1) exploration/initial visualization (notebooks ending with "exploration") (2) model development (notebooks ending with "modeling") and (3) an example extension \& application (the notebook [parametric_application](parametric_application.ipynb)). The specific material properties are:

1. Ultimate Tensile Stress:the ultimate stress sustained by a material during uniaxial testing, a measure of material strength.
2. Uniform Elongation: the (percent) elongation of the material at the Ultimate Tensile Stress measured during uniaxial testing, a measure of material ductility.
3. Thermal Conductivity: a measure of the material's ability to conduct heat.
4. Creep stress: a measure of the material's endurance to sustained loads over long periods of time.

### Goals
The overall goal is to assess the ability of various regression models to fit observed exerpimental data. The secondary goal is to use these models to assess the variablity of these material properties across material _alloys_, metals that have different compositions intended to enhance some desired material property. Finally, to develop an interpretable visualization procedure for _paramterized_ models that neatly allows me to assess "how good" an alloy needs to be for a specific application, see the below figure.

If you woud like to review the contents of this directory comprehensively, please follow this order:

1. [uts_data_exploration.ipynb](uts_data_expoloration.ipynb)
2. [uts_modeling.ipynb](uts_modeling.ipynb)
3. [ue_data_exploration.ipynb](ue_data_exploration.ipynb)
4. [ue_modeling.ipynb](ue_modeling.ipynb)
5. [conductivity_data_exploration.ipynb](conductivity_data_exploration.ipynb)
6. [conductivity_modeling.ipynb](conductivity_modeling.ipynb)
7. [creep_modeling.ipynb](creep_modeling.ipynb) 

![Parametric representation of hypothetical new material, compared with existing materials. _cf._ [parametric_application.ipynb](parametric_application.ipynb) for details](images/parametric_material_plot.svg)
