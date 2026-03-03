import wandb
from task_1_2 import train

# 1. Define the Sweep Configuration
sweep_config = {
    'method': 'bayes', # options: grid, random, bayes
    'metric': {
        'name': 'final_mAP50',
        'goal': 'maximize'   
    },
    'parameters': {
        'epochs':{
            'values': [50]
        },
        'batch_size': {
            'values': [16,32]
        },
        'learning_rate': {
            'distribution': 'log_uniform_values', 
            'min': 1e-5,
            'max': 1e-2
        },
        'lrf': {
            'distribution': 'log_uniform_values',
            'min': 0.001,
            'max': 0.2
        },
        'cos_lr': {
            'values': [True, False]
        },
        'optimizer': {
            'values': ['SGD', 'RMSProp', 'Adam', 'AdamW', 'NAdam']
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 0.1
        },
        'warmup_epochs': {
            'distribution': 'int_uniform',
            'min': 0,
            'max': 10
        },
        'img_size': {
            'values': [640]
        },
        'augment': {
            'values': [True, False]
        },
        'mosaic': {
            'distribution': 'uniform',
            'min': 0,
            'max': 1
        }
    }
}

# 2. Initialize the sweep
sweep_id = wandb.sweep(
    sweep_config, 
    project="C6-Week2",
)

# 3. Launch the agent
wandb.agent(sweep_id, function=train)