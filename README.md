# VK Project


## Installation and Setup

### Prerequisites

- Conda must be installed on your system. If you do not have Conda, please follow the installation instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

### Setting Up the Environment

1. **Clone the Repository**

   To get started, clone this repository to your local machine using the following command:

   ```bash
   git clone  https://github.com/olyandrevn/vk_project.git 
   cd vk_project
   ```
   
2. **Create and Activate the Conda Environment**

   This project uses a Conda environment to manage dependencies. Create the environment and activate it:
    
   ```bash
   conda env create -f environment.yml
   conda activate vk_proj_env
   ```

## Usage

### Training the Model
With the environment set up, you can proceed to train the model. Navigate to the project directory and run the training script:

```bash
python3 src/fit.py --data_path data/train_df.csv --model_path model/catboost_ranker_model.cbm
```

### Evaluating the Model
After training, evaluate the model's performance on the test dataset by running:

```bash
python3 src/evaluate.py --data_path data/test_df.csv --model_path model/catboost_ranker_model.cbm
```

## Results
For this test dataset, the following NDCG value was achieved:
```bash
NDCG value: 0.9362712791728599
```
