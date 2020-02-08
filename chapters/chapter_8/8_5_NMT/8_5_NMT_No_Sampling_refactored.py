from src.learner import Learner
from argparse import Namespace

args = Namespace(dataset_csv="data/nmt/simplest_eng_fra.csv",
                 vectorizer_file="vectorizer.json",
                 model_state_file="model.pth",
                 save_dir="model_storage/ch8/nmt_luong_no_sampling",
                 reload_from_files=False,
                 expand_filepaths_to_save_dir=True,
                 cuda=False,
                 seed=1337,
                 learning_rate=5e-4,
                 batch_size=64,
                 num_epochs=2,
                 early_stopping_criteria=5,
                 source_embedding_size=64,
                 target_embedding_size=64,
                 encoding_size=64,
                 catch_keyboard_interrupt=True,
                 sampling=False)


learner=Learner.learner_from_args(args)
learner.train()

learner.calc_bleu()
learner.plot_top_val_sentences(max_n=5)