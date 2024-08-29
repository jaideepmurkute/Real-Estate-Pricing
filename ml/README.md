# Machine Learning Training and Inference codebase - For development only
├── ml/                             
│   ├── models_store/               # Contains region-wise subdirs with saved trained models, scalers etc.
│   ├── training.py                 # ML model training and evaluation - local only
│   ├── prediction.py               # Evaluating the ML model - local only
|   |── generate_cols_to_use.py     # To generate dictionary of dates/columns to use - local only
|   |── generate_regionwise_data.py # Script to generate data files for each region - local only
│   ├── prediction.py               # To make predictions with the ML model
|   |── utils.py                    # General utility functions 
│   |── requirements.txt            
│   └── README.md                    