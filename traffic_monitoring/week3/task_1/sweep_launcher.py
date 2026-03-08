import wandb
#from task_1_2 import main
from traffic_monitoring.week3.task_2.task_2 import main

# 1. Define the Sweep Configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'hota_idf1',
        'goal': 'maximize'   
    },
    'parameters': {
        'iou_threshold': {
            'distribution': 'uniform',
            'min': 0.3,
            'max': 0.7
        },
        'conf_threshold': {
            'distribution': 'uniform',
            'min': 0.2,
            'max': 0.7
        },
        'max_age': {
            'distribution': 'int_uniform',
            'min': 0,
            'max': 55
        }
    }
}

ai_city_challenge = {
    'method': 'grid',
    'metric': {
        'name': 'hota_idf1',
        'goal': 'maximize'   
    },
    'parameters': {
        'cameras': {
            #'values': ['c001', 'c002', 'c003', 'c004', 'c005']
            'values': ['c010', 'c012', 'c013', 'c014', 'c015']
        }
    }
}

# 2. Initialize the sweep
sweep_id = wandb.sweep(
    ai_city_challenge, 
    project="C6-Week3",
)

# 3. Launch the agent
wandb.agent(sweep_id, function=main)