import pandas as pd
from utils import summarize, drop_multi_game_naive, map_lvgp_order, check_multi_game, feature_engineer
from config import *

cfg = CFG()

# ========= Process Train Data =============
df = pd.read_csv(ORIG_INPUT_PATH, usecols=cfg.COLS_TO_USE)
summarize(df, "X", 2)

df = drop_multi_game_naive(df)
df["lvgp_order"] = map_lvgp_order(df["level_group"])
if check_multi_game(df):
    print("There exist multiple game plays in at least one session.")

X = feature_engineer(df, train=True, cfg=cfg)
print(X.head())
X.to_parquet("./data/train_part.parquet")
