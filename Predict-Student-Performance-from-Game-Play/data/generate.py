import numpy as np
import pandas as pd


train_parquet = False
if train_parquet:
    train_df = pd.read_csv("./train.csv")
    train_df.drop(['fullscreen', 'hq', 'music'], inplace=True)
    train_df.to_parquet('./train.parquet')

labels_pre = True
if labels_pre:
    label_df = pd.read_csv("./train_labels.csv")
    QNS_PER_LV_GP = {"0-4": list(range(1, 4)), "5-12": list(range(4, 14)), "13-22": list(range(14, 19))}
    LVGP_ORDER = {"0-4": 0, "5-12": 1, "13-22": 2}

    y = label_df
    y["q"] = y["session_id"].apply(lambda x: x.split("_q")[1]).astype(int)
    y["session_id"] = y["session_id"].apply(lambda x: x.split("_q")[0]).astype('int64')
    y = y.sort_values(["session_id", "q"]).reset_index(drop=True)
    qn2lvgp = {qn: lv_gp for lv_gp, qns in QNS_PER_LV_GP.items() for qn in qns}
    y["level_group"] = y["q"].map(qn2lvgp)
    y_lvgp = y.groupby(["session_id", "level_group"]).apply(lambda x: list(x["correct"])).reset_index()
    y_lvgp["lvgp_order"] = y_lvgp["level_group"].map(LVGP_ORDER)
    y_lvgp = y_lvgp.sort_values(["session_id", "lvgp_order"]).reset_index(drop=True)
    y_lvgp.rename({0: "correct"}, axis=1, inplace=True)
    y_lvgp.drop(["lvgp_order"], axis=1, inplace=True)

    print("=====Labels Aggregated by `level_group`=====")
    print(y_lvgp.head(3))
    y_lvgp.to_parquet("../data/labels_lvgp.parquet")

    y_all = y_lvgp.groupby("session_id").apply(lambda x: list(x["correct"])).reset_index()
    ans_tmp = np.array(list(y_all.apply(lambda x: np.concatenate(x[0]), axis=1).values))
    y_all.drop(0, axis=1, inplace=True)
    y_all[list(range(18))] = ans_tmp
    y_all = y_all.set_index("session_id").sort_index()

    print("=====Flattened Labels=====")
    print(y_all.head(3))
    y_all.to_parquet("../data/labels_all.parquet")

