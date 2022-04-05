## Description
This repository contains some simple experiments to test the goodness of simple
attention mechanisms + [Time2Vec](https://arxiv.org/abs/1907.05321) on LSTM
networks.

Find more information in [this Quantdare post](www.quantdare.com).

## Run the experiments
Install the `requirements.txt` file and then run the `main.py` file:

```python
python main.py
```

## Hyperparameters and Feature Engineering
The hyperparameter configuration is only set to ensure all models have
approximately the same number of parameters.

`Beware of the number of parameters:` it must be approximately the same for all
models to ensure the comparison is fair.

Regarding the features, I haven't done anything since I just wanted to test the
models' goodness, not caring about achieving optimal performance.

## CREDITS
I did not implement the Time2Vec Module but I used the one create by
`garyzccisme` in [his repo](https://github.com/garyzccisme/Time2Vec/blob/main/time2vec.py).