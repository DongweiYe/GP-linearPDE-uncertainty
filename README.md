# **Gaussian Process for Learning PDEs with Uncertain Data Locations**

This repository provides the implementation and experimental results for the paper:

**PDE-Constrained Gaussian Process Surrogate Modeling with Uncertain Data Locations**  
*Dongwei Ye, Weihao Yan, Christoph Brune, Mengwu Guo*

---

## **Overview**

Gaussian Process (GP) regression is a state-of-the-art method for data-driven modeling and function approximation. This repository implements a Bayesian approach to GP regression that accounts for uncertainties in input data locations, addressing common challenges in learning partial differential equations (PDEs) from noisy and uncertain data. 

---

## **Installation**

1. Install required dependencies:
    ```
    pip install -r requirements.txt
    ```
---

## **Usage**

Run the script with the appropriate function key to reproduce the results:

```
python run.py <function_key>
```

### **Available Function Keys**
- **`h`**: Run experiments on the heat equation.
- **`rd`**: Run experiments on the reaction-diffusion equation.
- **`loadForPred`**: Load heat inference data for predictive analysis.
- **`loadForPred_rd`**: Load reaction-diffusion inference data for predictive analysis.

### **Examples for PDEs**

Run the heat equation experiment:
```
python run.py h
```

Load reaction-diffusion data for predictive analysis:
```
python run.py loadForPred_rd
```

### **Examples for 1D functions**
All the codes related to the 1D-function examples are located in the folder `1Dfunction`. Enter the directory and run the experiments with:
```
python run.py
```
Different 1D functions can be chosen for the test. The complete list of available functions is demonstrated in `1Dfunction/include/func.py`


