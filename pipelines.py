import torch
import numpy as np
from utils import Pipeline
from models import (
    AttentionLSTM,
    VanillaLSTM,
    EmbeddingLSTM,
    AttentionEmbeddingLSTM
)

torch.manual_seed(0)
np.random.seed(0)


class AttentionLSTMPipeline(Pipeline):
    def __init__(self, dataset="sine_wave"):
        super().__init__()

        self.dataset = dataset

    def create_model(self):
        self.generate_data()

        X_train = self.X_train
        y_train = self.y_train

        model = AttentionLSTM(
            embed_dim=X_train.shape[2], out_size=y_train.shape[-1]
        )
        self.model = model


class VanillaLSTMPipeline(Pipeline):
    def __init__(self, dataset="sine_wave"):
        super().__init__()

        self.dataset = dataset

    def create_model(self):
        self.generate_data()

        X_train = self.X_train
        y_train = self.y_train

        model = VanillaLSTM(
            input_size=X_train.shape[2], out_size=y_train.shape[-1]
        )
        self.model = model


class EmbeddingLSTMPipeline(Pipeline):
    def __init__(self, dataset="sine_wave"):
        super().__init__()

        self.dataset = dataset

    def create_model(self):
        self.generate_data()

        X_train = self.X_train
        y_train = self.y_train
        features = X_train.shape[1]
        mini_batch = X_train.shape[2]

        model = EmbeddingLSTM(
            linear_channel=features,
            period_channel=(mini_batch - features),
            input_channel=mini_batch,
            input_size=X_train.shape[2],
            out_size=y_train.shape[-1]
        )
        self.model = model


class AttentionEmbeddingLSTMPipeline(Pipeline):
    def __init__(self, dataset="sine_wave"):
        super().__init__()

        self.dataset = dataset

    def create_model(self):
        self.generate_data()

        X_train = self.X_train
        y_train = self.y_train
        features = X_train.shape[1]
        mini_batch = X_train.shape[2]

        model = AttentionEmbeddingLSTM(
            linear_channel=features,
            period_channel=(mini_batch - features),
            input_channel=mini_batch,
            input_size=X_train.shape[2],
            out_size=y_train.shape[-1]
        )
        self.model = model
