# Rouleax Morphology Classification

This repository entry contains the Jupyter notebook `KAN_Comparison.ipynb`, which analyzes and compares methods for classifying Rouleaux (rouleax) red blood cell morphology. The notebook reads and processes images from the Rouleax Red blood cells dataset, runs comparisons or experiments, and produces visualizations and evaluation outputs.

> Notebook permalink: https://github.com/YuvaChaithanya/Rouleax_Morphology_Classification/blob/71bd64855a9b45a61596c3adaf4e34e4e7a4c260/KAN_Comparison.ipynb

## Table of contents

- Overview
- Dataset
- What the notebook does
- Requirements
- How to run
  - Locally
  - On Kaggle
- Expected outputs
- Reproducibility tips
- Notes, caveats and known behaviors
- Contributing
- License & contact

## Overview

`KAN_Comparison.ipynb` inspects the Rouleax red blood cell images and performs a comparative analysis of classification / morphology analysis approaches (as implemented in the notebook). It prints dataset file listings, loads images, applies preprocessing and analysis steps, and visualizes results (example images, intermediate steps, metrics and plots). The notebook is organized into cells that walk through loading, preprocessing, model/algorithm evaluation and result summarization.

## Dataset

The notebook expects the Rouleax Red blood cell image dataset. In the environment where the notebook was originally executed, images were located under:

/kaggle/input/blood-cells/DATASET/Rouleax Red blood cells/

If you use the Kaggle dataset or mirror the files locally, point the notebook to the directory above or update the path variables inside the notebook.

## What the notebook does

Typical sections in the notebook include (may vary depending on the exact cell contents):

- Enumerate dataset files and sample images (the notebook prints many image paths).
- Load images and apply standard preprocessing (resizing, normalization, color-space conversions).
- Compute and compare features / models for Rouleax morphology classification.
- Visualize sample images, feature maps, intermediate processing, and evaluation metrics.
- Output tables or figures summarizing performance across methods.

Read and run the notebook to see specific algorithm names, exact preprocessing choices, and evaluation metrics used.

## Requirements

A typical Python environment for running the notebook should include:

- Python 3.8+ (or compatible)
- Jupyter / JupyterLab
- numpy
- pandas
- matplotlib
- seaborn
- scikit-image
- scikit-learn
- pillow (PIL)
- tqdm

If the notebook uses deep learning frameworks, you may need one of:
- TensorFlow >= 2.x or
- PyTorch >= 1.6

Install with pip if needed:

```bash
pip install numpy pandas matplotlib seaborn scikit-image scikit-learn pillow tqdm
# plus (if required) one of:
pip install tensorflow
# or
pip install torch torchvision
```

## How to run

### Locally (recommended)

1. Clone the repository or download the notebook.
2. Ensure the dataset is available locally and update the image path variables in the notebook if necessary.
3. Start Jupyter and open the notebook:

```bash
git clone https://github.com/YuvaChaithanya/Rouleax_Morphology_Classification.git
cd Rouleax_Morphology_Classification
jupyter notebook KAN_Comparison.ipynb
```

4. Run the cells in order (Kernel → Restart & Run All) to reproduce analyses and generate plots.

### On Kaggle

The notebook appears to have been executed on Kaggle with dataset mounted at `/kaggle/input/blood-cells/...`. To run on Kaggle:

1. Create a new Kaggle Notebook and add the dataset (if it is public on Kaggle).
2. Upload or open `KAN_Comparison.ipynb` in the Kaggle Notebook editor.
3. Run all cells. The stored image paths in the notebook should match the Kaggle input path.

You can also execute the notebook programmatically:

```bash
jupyter nbconvert --to notebook --execute KAN_Comparison.ipynb --output executed_notebook.ipynb
```

Note: Running the notebook headlessly may require adjusting plotting backends and paths.

## Expected outputs

- Printed listing of dataset file paths (the notebook prints many image file paths).
- Visualization figures showing sample Rouleaux images, processed images, feature visualizations and evaluation plots.
- Performance summaries (tables, metrics or printed statements) comparing approaches implemented in the notebook.

Because notebooks are exploratory, exact outputs depend on which sections are executed and whether any random seeds are set.

## Reproducibility tips

- Set random seeds (numpy, scikit-learn, frameworks) if you want deterministic results for models that use randomness.
- If the notebook depends on GPU/cuda for training, ensure the target environment has the necessary drivers and libraries.
- Verify that the image path variables point to the correct dataset location.

## Notes, caveats and known behaviors

- The notebook prints a very long list of image file paths when enumerating the dataset — this is expected if the dataset is large.
- Some cells may be marked with large outputs or heavy visualizations; running all cells in sequence may require time and resources depending on chosen algorithms.
- The notebook may contain commented or development code from experiments. Inspect each section before updating production code.

## Contributing

If you want to improve this notebook or make experiments reproducible:

- Add a requirements.txt or environment.yml describing exact package versions.
- Split long-running training/evaluation into separate cells or scripts.
- Add a small example subset and an automated script to run a minimal experiment for CI.

Pull requests welcome — please open an issue or PR in the repository.

## License & Contact

This project / notebook does not specify a license in the repository. If you plan to reuse or redistribute, please get clarification from the repository owner.

If you have questions, open an issue in the repository:
https://github.com/YuvaChaithanya/Rouleax_Morphology_Classification/issues

Acknowledgements: dataset contents and original notebook author.
