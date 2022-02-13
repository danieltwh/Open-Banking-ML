# Fintech Society ML - Open Banking

## Table of Contents

- [Fintech Society ML - Open Banking](#fintech-society-ml---open-banking)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Notes](#notes)
  - [Installation](#installation)
  - [Usage](#usage)

## Project Structure

```
.
├── data
│   └── raw
│   ├── intermediate
│   ├── processed
│   ├── temp
├── results
│   ├── outputs
│   ├── models
│   ├── weight
├── documents
│   ├── docs
│   ├── images
│   └── references
├── notebooks               <- notebooks for explorations / prototyping
│   ├── news
│   ├── signal
└── src                     <- all source code, internal org as needed
│   ├── train_lstm
│   ├── train_bert
```

## Notes

1. Model weights:
   Please download the following model weights and place them in the `weights` folder
   1. Link


## Installation

1. Clone this repo as follows

    ```bash
    git clone <THIS_REPO_SSH/HTTPS> 
    ```

2. Create the virutal environment
    
    ```bash
    conda create -n openbank python=3.7.12
    ```
    
3. Activate the virutal environment 
    
    ```bash
    conda activate openbank
    ``` 
   
4. Install the requirements by running

    ```bash
    python3 -m pip install -r requirements.txt
    ```

## Usage

### Demo

Video Demo of Application
