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

Run the main script with the appropriate function key to reproduce the results:

```
python main.py <function_key>
```

### **Available Function Keys**
- **`h`**: Run experiments on the heat equation.
- **`rd`**: Run experiments on the reaction-diffusion equation.
- **`loadForPred`**: Load heat inference data for predictive analysis.
- **`loadForPred_rd`**: Load reaction-diffusion inference data for predictive analysis.

### **Examples**

Run the heat equation experiment:
```
python main.py h
```

Load reaction-diffusion data for predictive analysis:
```
python main.py loadForPred_rd
```



