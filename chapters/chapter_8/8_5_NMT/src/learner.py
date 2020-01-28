import os
from pathlib import Path

import torch
from torch import optim
import torch.nn as nn
from tqdm import tqdm_notebook as tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from .dataset import NMTDataset, generate_nmt_batches
from .sampler import NMTSampler
from .model import NMTModel, NMTModelSampling
from .utils import set_seed_everywhere, handle_dirs
from .utils import make_train_state, compute_accuracy, sequence_loss
from .radam import RAdam


class Learner(object):
    def __init__(self, args, dataset, vectorizer, model):
        self.args = args
        self.dataset = dataset
        self.vectorizer = vectorizer
        self.model = model

        self.loss_func = sequence_loss
        self.optimizer = RAdam(self.model.parameters(), lr=args.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                              mode='min', factor=0.5,
                                                              patience=1)
        self.mask_index = vectorizer.target_vocab.mask_index
        self.train_state = make_train_state(args)

    def _set_splits_progress_bars(self):
        epoch_bar = tqdm(desc='training routine',
                         total=self.args.num_epochs,
                         position=0)

        self.dataset.set_split('train')
        train_bar = tqdm(desc='split=train',
                         total=self.dataset.get_num_batches(self.args.batch_size),
                         position=1,
                         leave=True)

        self.dataset.set_split('val')
        val_bar = tqdm(desc='split=val',
                       total=self.dataset.get_num_batches(self.args.batch_size),
                       position=1,
                       leave=True)
        return epoch_bar, train_bar, val_bar

    def _add_update_args(self, **kwargs):
        # turn into dict
        args_dict = vars(self.args)
        # changes also values in self.args
        for key, value in kwargs.items():
            args_dict[key] = value

    def train_eval_epoch(self, batch_generator, epoch_index, progress_bar, train_val='train'):
        if train_val not in ['train', 'val']:
            raise ValueError

        running_loss = 0.0
        running_acc = 0.0
        for batch_index, batch_dict in enumerate(batch_generator):
            # the training routine is these 5 steps:

            # --------------------------------------
            # step 1. zero the gradients
            self.optimizer.zero_grad()

            # step 2. compute the output
            y_pred = self.model(batch_dict['x_source'],
                                batch_dict['x_source_length'],
                                batch_dict['x_target'])

            # step 3. compute the loss
            loss = self.loss_func(y_pred, batch_dict['y_target'], self.mask_index)

            if train_val == 'train':
                # step 4. use loss to produce gradients
                loss.backward()

                # step 5. use optimizer to take gradient step
                self.optimizer.step()
            # -----------------------------------------
            # compute the running loss and running accuracy
            running_loss += (loss.item() - running_loss) / (batch_index + 1)
            # compute the accuracy
            acc_t = compute_accuracy(y_pred, batch_dict['y_target'], self.mask_index)
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            # update bar
            progress_bar.set_postfix(loss=running_loss, acc=running_acc,
                                     epoch=epoch_index)
            progress_bar.update()

        self.train_state[f'{train_val}_loss'].append(running_loss)
        self.train_state[f'{train_val}_acc'].append(running_acc)

    def train(self, **kwargs):
        # kwargs are meant to be training related arguments that might be changed for training
        self._add_update_args(**kwargs)

        epoch_bar, train_bar, val_bar = self._set_splits_progress_bars()
        try:
            for epoch_index in range(self.args.num_epochs):
                self.train_state['epoch_index'] = epoch_index

                # Iterate over training dataset
                # setup: batch generator, set loss and acc to 0, set train mode on
                self.dataset.set_split('train')
                batch_generator = generate_nmt_batches(self.dataset,
                                                       batch_size=self.args.batch_size,
                                                       device=self.args.device)
                self.model.train()
                self.train_eval_epoch(batch_generator, epoch_index, train_bar)
                # Iterate over val dataset
                # setup: batch generator, set loss and acc to 0; set eval mode on
                self.dataset.set_split('val')
                batch_generator = generate_nmt_batches(self.dataset,
                                                       batch_size=self.args.batch_size,
                                                       device=self.args.device)

                self.model.eval()
                self.train_eval_epoch(batch_generator, epoch_index, val_bar, 'val')
                # self.train_state = update_train_state(args=self.args, model=self.classifier,
                #                                     train_state=self.train_state)
                self.update_train_state()

                self.scheduler.step(self.train_state['val_loss'][-1])

                if self.train_state['stop_early']:
                    break

                train_bar.n = 0
                val_bar.n = 0
                epoch_bar.update()
        except KeyboardInterrupt:
            print("Exiting loop")

    def save_model(self):
        state = {
            'scheduler': self.scheduler,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_state': self.train_state,
            'args': self.args
        }
        torch.save(state, self.train_state['model_filename'])

    def load_model(self, filename):
        learner = torch.load(filename)
        self.scheduler = learner['scheduler']
        self.model.load_state_dict(learner['state_dict'])
        self.optimizer.load_state_dict(learner['optimizer'])
        self.train_state = learner['train_state']

    def update_train_state(self):
        """Handle the training state updates.

        Components:
         - Early Stopping: Prevent overfitting.
         - Model Checkpoint: Model is saved if the model is better

        :param args: main arguments
        :param model: model to train
        :param train_state: a dictionary representing the training state values
        :returns:
            a new train_state
        """

        # Save one model at least
        if self.train_state['epoch_index'] == 0:
            # torch.save(self.classifier.state_dict(), self.train_state['model_filename'])
            self.save_model()
            self.train_state['stop_early'] = False

        # Save model if performance improved
        elif self.train_state['epoch_index'] >= 1:
            loss_tm1, loss_t = self.train_state['val_loss'][-2:]

            # If loss worsened
            if loss_t >= self.train_state['early_stopping_best_val']:
                # Update step
                self.train_state['early_stopping_step'] += 1
            # Loss decreased
            else:
                # Save the best model
                if loss_t < self.train_state['early_stopping_best_val']:
                    self.save_model()
                    self.train_state['early_stopping_best_val'] = loss_t

                # Reset early stopping step
                self.train_state['early_stopping_step'] = 0

            # Stop early ?
            self.train_state['stop_early'] = \
                self.train_state['early_stopping_step'] >= self.args.early_stopping_criteria

    def calc_bleu(self):
        model = self.model.eval().to(self.args.device)

        sampler = NMTSampler(self.vectorizer, model)

        self.dataset.set_split('test')
        batch_generator = generate_nmt_batches(self.dataset,
                                               batch_size=self.args.batch_size,
                                               device=self.args.device)

        test_results = []
        for batch_dict in batch_generator:
            sampler.apply_to_batch(batch_dict)
            for i in range(self.args.batch_size):
                test_results.append(sampler.get_ith_item(i, False))

        plt.hist([r['bleu-4'] for r in test_results], bins=100)
        print(f"bleu-4 mean for test data: {np.mean([r['bleu-4'] for r in test_results])}")
        print(f"bleu-4 median for test data: {np.median([r['bleu-4'] for r in test_results])}")

    def get_batch_dict(self, dataset_name='val'):
        self.dataset.set_split(dataset_name)
        batch_generator = generate_nmt_batches(self.dataset,
                                               batch_size=self.args.batch_size,
                                               device=self.args.device)
        batch_dict = next(batch_generator)
        return batch_dict

    def plot_top_val_sentences(self, bleu_threshold=0.1, max_n=50):
        batch_dict = self.get_batch_dict()

        model = self.model.eval().to(self.args.device)
        sampler = NMTSampler(self.vectorizer, model)
        sampler.apply_to_batch(batch_dict)

        all_results = []
        for i in range(self.args.batch_size):
            all_results.append(sampler.get_ith_item(i, False))

        top_results = [x for x in all_results if x['bleu-4'] > bleu_threshold]
        print(f'sentence over threshold: {len(top_results)}')
        top_results = top_results[:max_n]

        for sample in top_results:
            plt.figure()
            target_len = len(sample['sampled'])
            source_len = len(sample['source'])

            attention_matrix = sample['attention'][:target_len, :source_len + 2].transpose()  # [::-1]
            ax = sns.heatmap(attention_matrix, center=0.0)
            ylabs = ["<BOS>"] + sample['source'] + ["<EOS>"]
            # ylabs = sample['source']
            # ylabs = ylabs[::-1]
            ax.set_yticklabels(ylabs, rotation=0)
            ax.set_xticklabels(sample['sampled'], rotation=90)
            ax.set_xlabel("Target Sentence")
            ax.set_ylabel("Source Sentence\n\n")

    def get_source_sentence(self, batch_dict, index):
        indices = batch_dict['x_source'][index].cpu().data.numpy()
        vocab = self.vectorizer.source_vocab
        return self.sentence_from_indices(indices, vocab)

    def get_true_sentence(self, batch_dict, index):
        return self.sentence_from_indices(batch_dict['y_target'].cpu().data.numpy()[index],
                                          self.vectorizer.target_vocab)

    def get_sampled_sentence(self, batch_dict, index):
        y_pred = self.model(x_source=batch_dict['x_source'],
                            x_source_lengths=batch_dict['x_source_length'],
                            target_sequence=batch_dict['x_target'])
        return self.sentence_from_indices(torch.max(y_pred, dim=2)[1].cpu().data.numpy()[index],
                                          self.vectorizer.target_vocab)

    def get_all_sentences(self, batch_dict, index):
        return {"source": self.get_source_sentence(batch_dict, index),
                "truth": self.get_true_sentence(batch_dict, index),
                "sampled": self.get_sampled_sentence(batch_dict, index)}

    def sentence_from_indices(self, indices, vocab, strict=True):

        out = []
        for index in indices:
            if index == vocab.begin_seq_index and strict:
                continue
            elif index == vocab.end_seq_index and strict:
                return " ".join(out)
            else:
                out.append(vocab.lookup_index(index))
        return " ".join(out)

    def get_val_1batch_sentence(self, index=1):
        batch_dict = self.get_batch_dict()
        results = self.get_all_sentences(batch_dict, index)
        return results

    @classmethod
    def learner_from_args(cls, args):
        if args.expand_filepaths_to_save_dir:
            args.vectorizer_file = os.path.join(args.save_dir,
                                                args.vectorizer_file)

            args.model_state_file = os.path.join(args.save_dir,
                                                 args.model_state_file)

            print("Expanded filepaths: ")
            print("\t{}".format(args.vectorizer_file))
            print("\t{}".format(args.model_state_file))

        # Check CUDA
        if not torch.cuda.is_available():
            args.cuda = False

        args.device = torch.device("cuda" if args.cuda else "cpu")
        print("Using CUDA: {}".format(args.cuda))

        # Set seed for reproducibility
        set_seed_everywhere(args.seed, args.cuda)

        # handle dirs
        handle_dirs(args.save_dir)

        if args.reload_from_files:
            # training from a checkpoint
            print("Loading dataset and loading vectorizer")
            dataset = NMTDataset.load_dataset_and_load_vectorizer(args.dataset_csv,
                                                                  args.vectorizer_file)
        else:
            # create dataset and vectorizer
            print("Loading dataset and creating vectorizer")
            dataset = NMTDataset.load_dataset_and_make_vectorizer(args.dataset_csv)
            dataset.save_vectorizer(args.vectorizer_file)
        vectorizer = dataset.get_vectorizer()

        if args.sampling:
            model_cls=NMTModelSampling
        else:
            model_cls=NMTModel
        model = model_cls(source_vocab_size=len(vectorizer.source_vocab),
                         source_embedding_size=args.source_embedding_size,
                         target_vocab_size=len(vectorizer.target_vocab),
                         target_embedding_size=args.target_embedding_size,
                         encoding_size=args.encoding_size,
                         target_bos_index=vectorizer.target_vocab.begin_seq_index)


        model = model.to(args.device)
        learner = cls(args, dataset, vectorizer, model)
        if args.reload_from_files:
            learner_states = torch.load(Path(args.model_state_file))
            learner.optimizer.load_state_dict(learner_states['optimizer'])
            learner.model.load_state_dict(learner_states['state_dict'])
            learner.scheduler = learner_states['scheduler']
            learner.train_state = learner_states['train_state']
        return learner
