from bert import ClassificationModel
import pandas as pd

# 使用验证集评估模型
# 读取训练数据并转换格式
with open("./asap_data/train.txt", "r", encoding="utf-8") as f:
    file = f.readlines()

train_data = []
for line in file:
    s,t = line.strip().split("\001")
    x = " ".join(t.split()[:-2]) + " [MASK] " + s
    label = t.split()[-2]
    train_data.append([x, label])

train_df = pd.DataFrame(train_data, columns=["text", "labels"])

# 将标签转换为整数
label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
train_df["labels"] = train_df["labels"].map(label_mapping)

# 设置模型参数
model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 300,
    "train_batch_size": 10,
    "num_train_epochs": 10,
    "output_dir": "./bert",
    "save_best_model":True,
    "evaluate_during_training": False,
    "use_multiprocessing_for_evaluation":False,
    "use_multiprocessing": False,
    "manual_seed": 42,
    "learning_rate": 1e-5,
}

# 初始化模型
modelcard = [
    ("roberta", "cardiffnlp/twitter-roberta-base-sentiment"), #9 epoch
    ("bert","bert-base-chinese"),
    ("roberta","clue/roberta_chinese_base")
]
model = ClassificationModel(
    *modelcard[2],
    num_labels=3,
    args=model_args,
)

# 训练模型
model.train_model(train_df)