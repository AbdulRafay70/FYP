import os
import sys
import json

sys.path.append(r"e:\FYP\training")
from data.split import create_train_val_test_split

dataset_dirs = [
    r"e:\FYP\datasets\BUSI\Dataset_BUSI_with_GT",
    r"e:\FYP\datasets\BUSBRA" # check if BUSBRA path exists, if not it will skip
]

# Filter out non-existent dirs
dataset_dirs = [d for d in dataset_dirs if os.path.exists(d)]
print(f"Loading datasets from: {dataset_dirs}")

splits = create_train_val_test_split(
    dataset_dirs=dataset_dirs,
    test_size=0.15,
    val_size=0.15,
    random_state=42
)
