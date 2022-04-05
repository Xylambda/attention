import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SequentialSampler
from torchfitter.trainer import Trainer
from torchfitter.utils.data import DataWrapper
from torchfitter.callbacks import (
    EarlyStopping,
    RichProgressBar,
    LearningRateScheduler
)
from datasets import (
    generate_sine_waves,
    generate_stock_returns,
    generate_white_noise,
    generate_venezia_high_waters,
)

torch.manual_seed(0)
np.random.seed(0)


class Time2Vec(nn.Module):
    """General Time2Vec Embedding/Encoding Layer.

    The input data should be related with timestamp/datetime.
        * Input shape: (*, feature_number), where * means any number of
        dimensions.
        * Output shape: (*, linear_channel + period_channel).

    Parameters
    ----------
    linear_channel : int
        The number of linear transformation elements.
    period_channel : int
        The number of cyclical/periodical transformation elements.
    input_channel : int
        The feature number of input data. Default = 1
    period_activation : int
        The activation function for periodical transformation. Default is sine
        function.

    References
    ----------
    .. [1] garyzccisme - Time2Vec
        https://github.com/garyzccisme/Time2Vec/blob/main/time2vec.py

    .. [2] Time2Vec: Learning a Vector Representation of Time
       https://arxiv.org/abs/1907.05321
    """
    def __init__(
        self,
        linear_channel: int,
        period_channel: int,
        input_channel: int = 1,
        period_activation=torch.sin
    ):
        super().__init__()
        
        self.linear_channel = linear_channel
        self.period_channel = period_channel
        
        self.linear_fc = nn.Linear(input_channel, linear_channel)
        self.period_fc = nn.Linear(input_channel, period_channel)
        self.period_activation = period_activation

    def forward(self, x):
        linear_vec = self.linear_fc(x)
        period_vec = self.period_activation(self.period_fc(x))
        return torch.cat([linear_vec, period_vec], dim=-1)


class Pipeline:
    """
    Class to ease the running of multiple experiments.
    """
    def __init__(self):
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.history = None
        self.y_pred = None

        self.preds = None
        self.tests = None

    def create_model(self):
        pass

    def generate_data(self):
        if self.dataset == "sine_wave":
            _tup = generate_sine_waves()
        
        elif self.dataset == "stock_returns":
            _tup = generate_stock_returns()

        elif self.dataset == "venezia":
            _tup = generate_venezia_high_waters()

        elif self.dataset == "white_noise":
            _tup = generate_white_noise()
        
        else:
            raise KeyError(f"Not supported dataset: {self.dataset}.")

        X_train, y_train, X_val, y_val, X_test, y_test = _tup

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

    def train_model(self):
        # ---------------------------------------------------------------------
        criterion = nn.HuberLoss()
        optimizer = optim.NAdam(self.model.parameters(), lr=0.005)
        sch = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.7, patience=20, min_lr=0.0001
        )

        # ---------------------------------------------------------------------
        callbacks = [
            EarlyStopping(patience=90, load_best=True),
            RichProgressBar(display_step=5, log_lr=True),
            LearningRateScheduler(scheduler=sch, on_train=False)
        ]

        # ---------------------------------------------------------------------
        train_wrapper = DataWrapper(
            self.X_train, self.y_train, dtype_X="float", dtype_y="float"
        )
        val_wrapper = DataWrapper(
            self.X_val, self.y_val, dtype_X="float", dtype_y="float"
        )
        train_loader = DataLoader(
            train_wrapper, batch_size=64, pin_memory=True
        )
        val_loader = DataLoader(val_wrapper, batch_size=64, pin_memory=True)

        # ---------------------------------------------------------------------
        trainer = Trainer(
            model=self.model,
            criterion=criterion,
            optimizer=optimizer,
            callbacks=callbacks,
            mixed_precision=True
        )

        # ---------------------------------------------------------------------
        history = trainer.fit(train_loader, val_loader, epochs=2000)
        self.history = history

        # to avoid memory problems
        test_wrapper = DataWrapper(
            self.X_test, self.y_test, dtype_X="float", dtype_y="float"
        )
        sampler = SequentialSampler(test_wrapper)
        test_loader = DataLoader(
            test_wrapper,
            batch_size=64,
            pin_memory=True,
            sampler=sampler,
            shuffle=False
        )

        y_pred = trainer.predict(test_loader, as_array=True)
        y_test = self.y_test

        # only works if predict horizon is 1
        preds = np.stack([row.flatten() for row in y_pred])
        tests = np.stack([row.flatten() for row in y_test])

        self.preds = preds
        self.tests = tests


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)