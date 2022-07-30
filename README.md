# Fintech Society ML - Open Banking

## Open Banking Project
In this project, we sought to develop Machine Learning (ML) models that predict future forex (FX) movements and sentiment of news headlines. The aim of the project was to create a one-stop-shop FX platform for investors and businesses to get the latest FX rates and news, as well as bidirectional price signals and news sentiment from our ML models to make better decisions on when to make Forex transactions.


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
│   ├── raw
│   ├── intermediate
│   ├── processed
│   └── temp
├── results
│   ├── outputs
│   ├── models
│   └── weights
├── documents
│   └── images
├── notebooks               <- notebooks for explorations / prototyping
│   ├── news
│   └── signal
└── src                     <- all source code, internal org as needed
```

## Notes

1. Model weights:
   Model weights are placed in the `weights` folder


## Installation

1. Clone this repo as follows

    ```bash
    git clone <THIS_REPO_SSH/HTTPS> 
    ```

2. Create the virutal environment
    
    ```bash
    conda create -n openbank python=3.7.11
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

### Website
This project was deployed on a website to show the Bidirectional Forex Signals from the LSTM model and the News Sentiment of Financial News Headlines with FinBERT. Please access the website [here](https://nus-fintech-open-banking.netlify.app/)




