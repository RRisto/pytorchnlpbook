import os

import torch
from torch import optim
import torch.nn as nn
from tqdm import tqdm_notebook, tqdm

from .dataset import SurnameDataset, generate_batches

from .utils import set_seed_everywhere, handle_dirs, compute_accuracy
from .classifier import SurnameClassifier
from .train import make_train_state, update_train_state
from .radam import RAdam


class Learner(object):
    def __init__(self, args, dataset, vectorizer, classifier):
        self.args = args
        self.dataset = dataset
        self.vectorizer = vectorizer
        self.classifier = classifier

        self.loss_func = nn.CrossEntropyLoss(weight=self.dataset.class_weights)
        # self.optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
        self.optimizer = RAdam(classifier.parameters(), lr=args.learning_rate)
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
            y_pred = self.classifier(batch_dict['x_surname'])

            # step 3. compute the loss
            loss = self.loss_func(y_pred, batch_dict['y_nationality'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            if train_val == 'train':
                # step 4. use loss to produce gradients
                loss.backward()

                # step 5. use optimizer to take gradient step
                self.optimizer.step()
            # -----------------------------------------
            # compute the accuracy
            acc_t = compute_accuracy(y_pred, batch_dict['y_nationality'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            # update bar
            progress_bar.set_postfix(loss=running_loss, acc=running_acc,
                                     epoch=epoch_index)
            progress_bar.update()

        self.train_state[f'{train_val}_loss'].append(running_loss)
        self.train_state[f'{train_val}_acc'].append(running_acc)

    def train(self, **kwargs):
        #kwargs are meant to be trianing related arguments that might be changed for training
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
                self.classifier.train()
                self.train_eval_epoch(batch_generator, epoch_index, train_bar)
                # Iterate over val dataset
                # setup: batch generator, set loss and acc to 0; set eval mode on
                self.dataset.set_split('val')
                batch_generator = generate_batches(self.dataset,
                                                   batch_size=self.args.batch_size,
                                                   device=self.args.device)

                self.classifier.eval()
                self.train_eval_epoch(batch_generator, epoch_index, val_bar, 'val')
                train_state = update_train_state(args=self.args, model=self.classifier,
                                                 train_state=self.train_state)

                self.scheduler.step(train_state['val_loss'][-1])

                if train_state['stop_early']:
                    break

                train_bar.n = 0
                val_bar.n = 0
                epoch_bar.update()
        except KeyboardInterrupt:
            print("Exiting loop")

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
            dataset = SurnameDataset.load_dataset_and_load_vectorizer(args.surname_csv,
                                                                      args.vectorizer_file)
        else:
            # create dataset and vectorizer
            dataset = SurnameDataset.load_dataset_and_make_vectorizer(args.surname_csv)
            dataset.save_vectorizer(args.vectorizer_file)
        vectorizer = dataset.get_vectorizer()

        classifier = SurnameClassifier(initial_num_channels=len(vectorizer.surname_vocab),
                                       num_classes=len(vectorizer.nationality_vocab),
                                       num_channels=args.num_channels)
        classifier = classifier.to(args.device)
        dataset.class_weights = dataset.class_weights.to(args.device)
        return cls(args, dataset, vectorizer, classifier)
