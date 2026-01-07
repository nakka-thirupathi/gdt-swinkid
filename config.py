import torch


class Config:
    # === DEVICE & REPRODUCIBILITY === #
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42

    # === DATA CONFIGURATION === #
    IMG_SIZE = 224
    LABELS = ["Normal", "Cyst", "Tumor", "Stone"]
    NUM_CLASSES = 4
    NUM_WORKERS = 2
    TRAIN_SPLIT = 0.8

    # === MODEL CONFIGURATION === #
    INPUT_DIM = 768        # CvT feature dimension
    LATENT_DIM = 64        # Autoencoder latent dimension

    # === TRAINING HYPERPARAMETERS === #
    BATCH_SIZE = 32
    AE_BATCH_SIZE = 128
    AE_EPOCHS = 50
    AE_LEARNING_RATE = 0.001

    # === XGBOOST CONFIGURATION (MULTI-CLASS) === #
    XGB_PARAMS = {
        "objective": "multi:softprob",
        "num_class": NUM_CLASSES,
        "eval_metric": ["mlogloss", "merror"],
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "random_state": SEED,
        "tree_method": "hist",
    }

    # === PATHS === #
    MODELS_DIR = "models"
    RESULTS_DIR = "results"
    CHECKPOINTS_DIR = "checkpoints"
