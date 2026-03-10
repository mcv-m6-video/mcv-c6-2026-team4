import wandb
from task_2 import main

ai_city_challenge = {
    'method': 'grid',
    'metric': {
        'name': 'hota_idf1',
        'goal': 'maximize'   
    },
    'parameters': {
        'camera': {
            #'values': ['c001', 'c002', 'c003', 'c004', 'c005']
            #'values': ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
            'values': ['c016', 'c017', 'c018', 'c019', 'c020', 'c021', 'c022', 'c023', 'c024', 'c025', 'c026', 'c027', 'c028']
        },
        'sequence': {
            #'values': ['S01']
            #'values': ['S03']
            'values': ['S04']
        }
    }
}

# 2. Initialize the sweep
sweep_id = wandb.sweep(
    ai_city_challenge, 
    project="C6-Week3-Task2-Definitiu",
)

# 3. Launch the agent
wandb.agent(sweep_id, function=main)