from seq2seq_model_M import Seq2SeqModel
import pandas as pd
# logging.basicConfig(level=logging.INFO)
# transformers_logger = logging.getLogger("transformers")
# transformers_logger.setLevel(logging.WARNING)


with open("./asap_data/train.txt", "r", encoding="utf-8") as f:
    file = f.readlines()
train_data = []
for line in file:
    x, y = line.split("\001")[0], line.strip().split("\001")[1]
    train_data.append([x, y])

# train_data = [
#     ["one", "1"],
#     ["two", "2"],
# ]

train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

# steps = [1, 2, 3, 4, 6]
# learing_rates = [4e-5, 2e-5, 1e-5, 3e-5]
steps = [1]
learing_rates = [4e-5]



best_accuracy = 0
for lr in learing_rates:
    for step in steps:
        model_args = {
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "output_dir": "./t5",
            "max_seq_length": 300,
            "train_batch_size": 16,
            "num_train_epochs": 10,
            "save_best_model":True,
            "save_eval_checkpoints": False,
            "save_model_every_epoch": False,
            "evaluate_during_training": False,
            "evaluate_generated_text": False,
            "evaluate_during_training_verbose": False,
            "use_multiprocessing": False,
            "max_length": 300,
            "manual_seed": 42,
            "gradient_accumulation_steps": step,
            "learning_rate":  lr,
            "save_steps": 99999999999999,
        }

        model = Seq2SeqModel(
            encoder_decoder_type="t5",
            encoder_decoder_name="uer/t5-small-chinese-cluecorpussmall",
            args=model_args,
        )

        best_accuracy = model.train_model(train_df, best_accuracy)