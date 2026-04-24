from datetime import datetime
from datasets import load_dataset
from utils.trainer import Trainer, TrainerConfig
from utils.seed import set_seed
from utils.dataset import CodeDataset
from utils.model import MARGINModel
from utils.logger import log


def main():
    TIME_PREFIX = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # 定义数据集和模型名称
    DATASET_NAME = "codemetic/MARGIN"
    DATASET_SUBSET = "debug"
    BACKBONE_NAME = "microsoft/unixcoder-base"

    # 超参数
    BASE_SCALE = 10
    CONFIDENCE_ALPHA = 0.95

    # 配置
    SEED = 42
    DEVICE = "cuda:1"

    config = TrainerConfig(
        output_dir=f"./output/{DATASET_SUBSET}-{BACKBONE_NAME.split('/')[1]}-{TIME_PREFIX}",
        batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        max_epochs=200,
        early_stopping_patience=200,
        device=DEVICE,
        umap_n_neighbors=15,
        umap_min_dist=0.1,
        seed=SEED,
    )

    set_seed(config.seed)

    log.print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET)

    train_hf = dataset["train"]
    val_hf = dataset["val"]
    test_hf = dataset["test"]

    log.print(
        f"Train size: {len(train_hf)}, Val size: {len(val_hf)}, Test size: {len(test_hf)}"
    )

    log.print(f"Loading tokenizer and model: {BACKBONE_NAME}")

    train_dataset = CodeDataset(BACKBONE_NAME, train_hf)
    # val_dataset = CodeDataset(MODEL_NAME, val_hf, config.max_length)
    test_dataset = CodeDataset(BACKBONE_NAME, test_hf)

    label2id = train_dataset.label2idx

    log.print(f"Number of classes: {len(label2id)}")
    log.print(f"Label mapping: {label2id}")

    model = MARGINModel(
        backbone=BACKBONE_NAME,
        base_scale=BASE_SCALE,
        alpha=CONFIDENCE_ALPHA,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )

    # 训练
    trainer = Trainer(model, config)
    trainer.train(DATASET_SUBSET, BACKBONE_NAME)


if __name__ == "__main__":
    main()
