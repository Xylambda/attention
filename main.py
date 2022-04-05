import logging
import numpy as np
from pathlib import Path
from pipelines import (
    AttentionLSTMPipeline,
    VanillaLSTMPipeline,
    EmbeddingLSTMPipeline,
    AttentionEmbeddingLSTMPipeline
)
from utils import count_parameters
from torchfitter.io import save_pickle
from torchfitter.utils.convenience import get_logger


RESULTS_PATH = Path("results")


logger = get_logger(name="Experiments")
level = logger.level
logging.basicConfig(level=level)


if __name__ == "__main__":
    datasets = ["sine_wave", "white_noise", "venezia", "stock_returns"]
    pipelines = [
        VanillaLSTMPipeline,
        AttentionLSTMPipeline,
        EmbeddingLSTMPipeline,
        AttentionEmbeddingLSTMPipeline
    ]

    for key in datasets:

        logger.info(f"PROCESSING DATASET: {key}")
        folder = RESULTS_PATH / f"{key}"
        folder.mkdir(exist_ok=True)

        for _pipe in pipelines:
            pipe = _pipe(dataset=key)
            pip_name = _pipe.__name__

            logger.info(f"TRAINING: {pip_name}")

            pipe.create_model()

            logger.info(f"NUMBER OF PARAMS: {count_parameters(pipe.model)}")

            pipe.train_model()

            y_pred = pipe.preds
            y_test = pipe.tests
            history = pipe.history

            pipe_folder = folder / f"{pip_name}"
            pipe_folder.mkdir(exist_ok=True)

            np.save(file=pipe_folder / "y_pred", arr=y_pred)
            np.save(file=pipe_folder / "y_test", arr=y_test)

            save_pickle(obj=history, path=pipe_folder / "history.pkl")
