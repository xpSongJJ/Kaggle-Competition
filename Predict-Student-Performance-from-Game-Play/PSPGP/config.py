
N_QNS = 18
LEVEL = list(range(23))
LEVEL_GROUP = ["0-4", "5-12", "13-22"]
LVGP_ORDER = {"0-4": 0, "5-12": 1, "13-22": 2}
QNS_PER_LV_GP = {"0-4": list(range(1, 4)), "5-12": list(range(4, 14)), "13-22": list(range(14, 19))}
LV_PER_LV_GP = {"0-4": list(range(0, 5)), "5-12": list(range(5, 13)), "13-22": list(range(13, 23))}
CAT_FEAT_SIZE = {
    "event_comb_code": 19,
    "room_fqid_code": 19,
}
ORIG_INPUT_PATH = "../data/train.csv"
TARGET_PATH = "../data/train_labels.csv"


class CFG:
    # ==Mode==
    # Specify True to enable model training
    train = True

    # ==Data===
    FEATS = ["et_diff", "event_comb_code", "room_fqid_code"]
    CAT_FEATS = ["event_comb_code", "room_fqid_code"]
    COLS_TO_USE = ["session_id", "level", "level_group", "elapsed_time",
                   "event_name", "name", "room_fqid"]
    T_WINDOW = 1000

    # ==Training==
    SEED = 42
    DEVICE = "cuda:0"
    EPOCH = 100
    CKPT_METRIC = "f1@0.63"

    # ==DataLoader==
    BATCH_SIZE = 256
    NUM_WORKERS: 4

    # ==Solver==
    LR = 1e-3
    WEIGHT_DECAY = 1e-4

    # ==Early Stopping==
    ES_PATIENCE = 0

    # ==Evaluator==
    EVAL_METRICS = ["auroc", "f1"]