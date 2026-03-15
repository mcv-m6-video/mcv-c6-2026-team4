import pandas as pd
from src.eval import readData, eval as aicity_eval, print_results, get_results

# 1. Read your generated predictions
pred_file = "mtmc_predictions.txt"
df = pd.read_csv(pred_file, header=None)

"""# 2. Force the bounding box coordinates (columns 3, 4, 5, 6) to integers
df[3] = df[3].astype(int)
df[4] = df[4].astype(int)
df[5] = df[5].astype(int)
df[6] = df[6].astype(int)

# 3. Save the fixed version
fixed_file = "mtmc_predictions_fixed.txt"
df.to_csv(fixed_file, header=None, index=False)
print("Fixed floating point coordinates in predictions.")"""

# 4. Run the official evaluation
GLOBAL_GT_PATH = "../data/AI_CITY_CHALLENGE_2022_TRAIN/eval/ground_truth_train.txt"
AI_CITY_BASE_PATH = "../data/AI_CITY_CHALLENGE_2022_TRAIN"

test_df = readData(GLOBAL_GT_PATH)
pred_df = readData(pred_file)

summary = aicity_eval(test_df, pred_df, mread=False, dstype="train/S01", roidir=AI_CITY_BASE_PATH)
print_results(summary, mread=False)
results=get_results(summary)
print(results)