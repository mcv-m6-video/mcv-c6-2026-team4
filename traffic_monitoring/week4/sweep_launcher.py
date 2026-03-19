import wandb
from mtmc import main

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'hota',
        'goal': 'maximize'   
    },
    'parameters': {
        'iou_threshold': {
            'distribution': 'uniform',
            'min': 0.15,
            'max': 0.7
        },
        'conf_threshold': { 
            'distribution': 'uniform',
            'min': 0.4,
            'max': 0.7
        },
        'max_age': {
            'distribution': 'int_uniform',
            'min': 0, 
            'max': 20
        },
        'global_match_threshold': {
            'distribution': 'uniform', 
            'min': 0.5,
            'max': 0.9
        }
    }
}

# 2. Initialize the sweep
sweep_id = wandb.sweep(
    sweep_config, 
    project="C6-Week4",
)

# 3. Launch the agent
wandb.agent(sweep_id, function=main)