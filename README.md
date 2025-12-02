# Fused Geometric-Machine Leaning Model for Transparent Substrate Placement Detection

![./model_workflows_R1.png](./model_workflows_R1.png)

**Authors:** Kelsey Fontenot kelfon@mit.edu and Anjali Gorti agorti@mit.edu

## Overview
Self-driving laboratories (SDLs) are beginning to aid the chemistry and materials discovery process by automating time-consuming and repetitive tasks. Though many SDLs are emerging, there does not yet exist a methodology for handling delicate and transparent substrates for such experiments. Toward this end, we develop we propose a method of Automated Substrate Handling and Exchange (ASHE) for transparent substrates within SDLs. ASHE utilizes a robotic arm with custom designed grippers, a dual-actuated substrate dispenser, and a fused geometric and deep learning vision detection model to accurately unload used substrates and load fresh substrates fully automatically within a self-driving laboratory. In 130 independent trials of ASHE reloading substrates into a self-driving laboratory, the systems demonstrates 98.5\% placement accuracy with only two substrate misplacements. ASHE automatically detects these misplaced substrates and corrects their placements. Although ASHE demonstrates promising performance results towards the advancement of automated transparent substrate manipulation, vision detection, and error correction, several limitations exist in its cost and its generalizability as the system is heavily designed around substrates of specific sizes. Despite these limitations, ASHE helps to close the research gap in fragile and transparent substrate manipulation for self-driving materials laboratories.

## Repository Structure

| File/Folder               | Description                                                                                                                                    |
|---------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| [examples.ipynb](./examples.ipynb)         | Jupyter notebook demonstrating the pipeline and usage examples of the fused geometric-machine learning model for transparent substrate detection.|
| [cnn_model.py](./cnn.py)                | Python module with the machine learning model.       |
| [geometric_model.py](./geometric.py)                | Python module with the geometric model.                            |
| [standard_test.xlsx](./standard_test.xlsx)    | Results from the full 130 placement ASHE run. |
| [test_images](./test_images) | Folder containing test images to demonstrate models.|


## Requirements
To run the code in this repository, you will need the following dependencies:

- python=3.11.11
- torch==2.5.1
- opencv-python==4.11.0.86
- shapely==2.1.0
- pyrealsense2==2.55.1.6486
- numpy==2.0.1
- torchaudio==2.5.1
- torchvision==0.20.1
- pillow==11.1.0

## Installation

1. Download Anaconda Navigator & open the prompt terminal.
2. In the terminal, create a new virtual environment by entering:
```
conda create -n <env-name>
```
3. Then, activate the environment:
```
conda activate <env-name>
```
4. In the environment, install the correct version of python with conda. Then, install the rest of the dependencies with pip:
```
conda install python=3.11.11
pip install -r requirements.txt
```

## Quick Start

## Model Architecture

## Results
