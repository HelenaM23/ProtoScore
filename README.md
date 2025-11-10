# Protoscore: A Framework for Evaluating Prototype-Based XAI

This repository contains code for training and evaluating prototype-based explainable AI models using the **ProtoScore** framework, which provides a comprehensive assessment of prototype quality across multiple dimensions.

## Overview

The system supports the training and evaluation of prototype-based XAI methods especially on time series datasets. It implements the ProtoScore framework described in ["From Confusion to Clarity: ProtoScore - A Framework for Evaluating Prototype-Based XAI" (FAccT 2025)](https://dl.acm.org/doi/full/10.1145/3715275.3732151), which evaluates prototype quality using adapted Co-12 properties from Nauta et al. [1].

## Features

- **Multiple Prototype Methods**: Examples include [MAP](https://proceedings.mlr.press/v189/obermair23a.html) (Model-Agnostic Prototype), [MSP](http://dblp.uni-trier.de/db/conf/ijcai/khd2019.html) (Model-Specific Prototype), and [EBE method](https://doi.org/10.1007/978-3-031-91398-3_32)
- **Comprehensive Evaluation**: Evaluates 9 properties using quantitative evaluation metrics including Correctness, Consistency, Continuity, Contrastivity, Covariate Complexity, Compactness, Confidence, Input Completeness, Cohesion of Latent Space
- **Multiple Datasets**: Pre-configured support for [ECG200](https://www.timeseriesclassification.com/description.php?Dataset=ECG200), [FordA](https://www.timeseriesclassification.com/description.php?Dataset=FordA)/[FordB](https://www.timeseriesclassification.com/description.php?Dataset=FordB), [StarLightCurve](https://www.timeseriesclassification.com/description.php?Dataset=StarlightCurves), [Wafer](https://www.timeseriesclassification.com/description.php?Dataset=Wafer), and [SAWSINE datasets](https://proceedings.mlr.press/v189/obermair23a.html)
- **Configurable**: Uses [Hydra](https://github.com/facebookresearch/hydra)[2] for flexible configuration management


## Installation

Dependencies:
    Python >=3.12

```bash
pip install -r requirements.txt
pip install -e .
```

or using Makefile

```bash
make install
```

## Usage

### Training Models
![UML Diagram](uml/uml_training.puml)
```bash
python -m scripts.train_prototypes
```
or 
```bash
make train
```

- Training configurations are managed via `config/train_hyperparameter.yaml` with subfolders `method` and `dataset`.
- Trained models, their statistics and visualization of the prototypes are saved in `results/model`. 

### Evaluating Models
![UML Diagram](uml/uml_evaluation_all.puml)

```bash
python -m scripts.eval_prototypes
```
or
```bash
make eval
```

- Training configurations are managed via `config/eval_hyperparameter.yaml` with subfolders `method`, `dataset`, `evaluation_metrics`, `clustering_config`, `metric_weights`, and `plot_config`.
- Comprehensive evaluation reports and latent space visualization are saved in `results/evaluation`.

## Integrated Methods to be evaluated

- **MAP (Model-Agnostic Prototype)**: Christoph Obermair, Alexander Fuchs, Franz Pernkopf, Lukas Felsberger, Andrea
Apollonio, and DanielWollmann. 2023. Example or Prototype? Learning Concept-
Based Explanations in Time-Series. Proceedings of The 14th Asian Conference on
Machine Learning 189 (12–14 Dec 2023), 816–831. https://proceedings.mlr.press/
v189/obermair23a.html
- **MSP (Model-Specific Prototype)**: Alan H Gee, Diego Garcia-Olano, Joydeep Ghosh, and David Paydarfar. 2019.
Explaining Deep Classification of Time-Series Data with Learned Prototypes.
CEUR workshop proceedings 2429 (2019), 15–22. http://dblp.uni-trier.de/db/conf/
ijcai/khd2019.html
- **EBE (Example-Based Explanations)**: Genghua Dong, Henrik Boström, Michalis Vazirgiannis, and Roman Bresson. 2025. Obtaining Example-Based Explanations from Deep Neural Networks. In Advances in Intelligent Data Analysis XXIII: 23rd International Symposium on Intelligent Data Analysis, IDA 2025, Konstanz, Germany, May 7–9, 2025, Proceedings. Springer-Verlag, Berlin, Heidelberg, 432–443. https://doi.org/10.1007/978-3-031-91398-3_32

## Supported Datasets


- [**ECG200**](https://www.timeseriesclassification.com/description.php?Dataset=ECG200): Cardiac activity classification (Normal/Abnormal)
- [**FordA**](https://www.timeseriesclassification.com/description.php?Dataset=FordA)/[**FordB**](https://www.timeseriesclassification.com/description.php?Dataset=FordB): Engine noise fault detection

- [**StarLightCurve**](https://www.timeseriesclassification.com/description.php?Dataset=StarlightCurves): Astronomical light curve classification (3 classes)
- [**Wafer**](https://www.timeseriesclassification.com/description.php?Dataset=Wafer): Semiconductor manufacturing defect detection

- [**SAWSINE**](https://proceedings.mlr.press/v189/obermair23a.html): Synthetic time series with noise

## ProtoScore Evaluation Metrics

The framework evaluates prototypes across 9 key properties, based on the Co-12-properties of Nauta et al. [1]:

| Property | Description |
|----------|-------------|
| **Correctness (CR)** | How accurately prototypes reflect model reasoning |
| **Consistency (CS)** | Stability across different training runs |
| **Continuity (CN)** | Robustness to input perturbations |
| **Contrastivity (CT)** | Ability to distinguish between classes |
| **Covariate Complexity (CC)** | Interpretability of prototype features |
| **Compactness (CP)** | Appropriate number of prototypes |
| **Confidence (CF)** | Certainty in prototype assignments |
| **Input Completeness (IC)** | Coverage of the data distribution |
| **Cohesion of Latent Space (CLS)** | Quality of latent space clustering |

## Integrate your own methods or datasets

You can also include your own prototype methods or datasets into the framework. 

1. Add your method: Use the structure in `src/methods/`. Implement necessary wrappers and register your method in the factory classes.
2. Add your dataset: Implement a new dataset class in `src/core/data.py` and register it in the `DatasetFactory` class.
3. Update configuration: Add your method/dataset to the relevant config files in `config/`.



## References
[1] Nauta, Meike; Trienes, Jan; Pathak, Shreyasi; Nguyen, Elisa; Peters, Michelle; Schmitt, Yasmin; Schlötterer, Jörg; van Keulen, Maurice; Seifert, Christin (2023): From Anecdotal Evidence to Quantitative Evaluation Methods: A Systematic Review on Evaluating Explainable AI. ACM Computing Surveys, Vol. 55, No. 13s, Article 295, pp. 1–42. Association for Computing Machinery, New York, NY, USA. https://doi.org/10.1145/3583558

[2] Yadan, Omry (2019): Hydra – A framework for elegantly configuring complex applications. Github. https://github.com/facebookresearch/hydra

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{monke2025protoscore,
  title={From Confusion to Clarity: ProtoScore - A Framework for Evaluating Prototype-Based XAI},
  author={Monke, Helena and Sae-Chew, Benjamin and Fresz, Benjamin and Huber, Marco F.},
  booktitle={The 2025 ACM Conference on Fairness, Accountability, and Transparency},
  publisher = {Association for Computing Machinery},
  year={2025},
  url = {https://doi.org/10.1145/3715275.3732151},
  doi = {10.1145/3715275.3732151}
}
```