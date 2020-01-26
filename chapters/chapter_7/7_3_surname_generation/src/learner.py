import os
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm_notebook as tqdm
from functools import partial

from .dataset import generate_batches, SurnameDataset
from .utils import set_seed_everywhere, handle_dirs, make_train_state, sequence_loss
from .model import SurnameGenerationModel, sample_from_model, decode_samples, SurnameGenerationModelConditioned, \
    sample_from_model_conditioned
from .utils import compute_accuracy
from .radam import RAdam


class Learner(object):
    def __init__(self, args, dataset, vectorizer, model):
        self.args = args
        self.dataset = dataset
        self.vectorizer = vectorizer
        self.model = model
        self.mask_index = self.vectorizer.char_vocab.mask_index

        self.loss_func = partial(sequence_loss, mask_index=self.mask_index)
        self.optimizer = RAdam(model.parameters(), lr=args.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                              mode='min', factor=0.5,
                                                              patience=1)
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
            if self.args.conditioned:
                y_pred = self.model(x_in=batch_dict['x_data'], nationality_index=batch_dict['class_index'])
            else:
                y_pred = self.model(x_in=batch_dict['x_data'])

            # step 3. compute the loss
            loss = self.loss_func(y_pred, batch_dict['y_target'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            if train_val == 'train':
                # step 4. use loss to produce gradients
                loss.backward()

                # step 5. use optimizer to take gradient step
                self.optimizer.step()
            # -----------------------------------------
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
                batch_generator = generate_batches(self.dataset,
                                                   batch_size=self.args.batch_size,
                                                   device=self.args.device)
                self.model.train()
                self.train_eval_epoch(batch_generator, epoch_index, train_bar)
                # Iterate over val dataset
                # setup: batch generator, set loss and acc to 0; set eval mode on
                self.dataset.set_split('val')
                batch_generator = generate_batches(self.dataset,
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

    def validate(self):
        self.load_model(self.train_state['model_filename'])
        self.model = self.model.to(self.args.device)

        self.dataset.set_split('test')
        batch_generator = generate_batches(self.dataset,
                                           batch_size=self.args.batch_size,
                                           shuffle=False,
                                           device=self.args.device)
        running_loss = 0.
        running_acc = 0.
        self.model.eval()

        for batch_index, batch_dict in enumerate(batch_generator):
            # compute the output
            if self.args.conditioned:
                y_pred = self.model(batch_dict['x_data'], nationality_index=batch_dict['class_index'])
            else:
                y_pred = self.model(batch_dict['x_data'])

            # compute the loss
            loss = self.loss_func(y_pred, batch_dict['y_target'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # compute the accuracy
            acc_t = compute_accuracy(y_pred, batch_dict['y_target'], self.mask_index)
            running_acc += (acc_t - running_acc) / (batch_index + 1)

        self.train_state['test_loss'] = running_loss
        self.train_state['test_acc'] = running_acc

        print(f"Test loss: {round(self.train_state['test_loss'], 3)}")
        print(f"Test Accuracy: {round(self.train_state['test_acc'], 3)}")

    def generate_unconditoned(self, num_names):
        # Generate nationality hidden state
        sampled_surnames = decode_samples(
            sample_from_model(self.model, self.vectorizer, num_samples=num_names),
            self.vectorizer)
        # Show results
        print("-" * 15)
        for i in range(num_names):
            print(sampled_surnames[i])

    def generate_conditioned(self, num_names):
        for index in range(len(self.vectorizer.nationality_vocab)):
            nationality = self.vectorizer.nationality_vocab.lookup_index(index)
            print("Sampled for {}: ".format(nationality))
            sampled_indices = sample_from_model_conditioned(self.model, self.vectorizer,
                                                            nationalities=[index] * num_names, temperature=0.7)
            for sampled_surname in decode_samples(sampled_indices, self.vectorizer):
                print("-  " + sampled_surname)

    def generate(self, num_names):
        if self.args.conditioned:
            self.generate_conditioned(num_names)
        else:
            self.generate_unconditoned(num_names)

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
            dataset = SurnameDataset.load_dataset_and_load_vectorizer(args.surname_csv,
                                                                      args.vectorizer_file)
        else:
            # create dataset and vectorizer
            print("Loading dataset and creating vectorizer")
            dataset = SurnameDataset.load_dataset_and_make_vectorizer(args.surname_csv)
            dataset.save_vectorizer(args.vectorizer_file)
        vectorizer = dataset.get_vectorizer()

        if args.conditioned:
            model = SurnameGenerationModelConditioned(char_embedding_size=args.char_embedding_size,
                                                      char_vocab_size=len(vectorizer.char_vocab),
                                                      num_nationalities=len(vectorizer.nationality_vocab),
                                                      rnn_hidden_size=args.rnn_hidden_size,
                                                      padding_idx=vectorizer.char_vocab.mask_index,
                                                      dropout_p=0.5)
        else:
            model = SurnameGenerationModel(char_embedding_size=args.char_embedding_size,
                                           char_vocab_size=len(vectorizer.char_vocab),
                                           rnn_hidden_size=args.rnn_hidden_size,
                                           padding_idx=vectorizer.char_vocab.mask_index)

        model = model.to(args.device)
        learner = cls(args, dataset, vectorizer, model)
        if args.reload_from_files:
            learner_states = torch.load(Path(args.model_state_file))
            learner.optimizer.load_state_dict(learner_states['optimizer'])
            learner.model.load_state_dict(learner_states['state_dict'])
            learner.scheduler = learner_states['scheduler']
            learner.train_state = learner_states['train_state']
        return learner
