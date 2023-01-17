# OpenPack-Challenge-1st
The 1st place solution of [OpenPack Challenge 2022](https://open-pack.github.io/challenge2022).

## Main program

- src/train.py
- src/utils/model.py
- src/ensemble_mean.py

## How to reproduce submission results
1. `$ optk-download -d ./data`
1. Run `src/train.py` 5 times with configs under `configs/submission`.
    - Change `config_name` in `@hydra.main`
1. Run `src/ensemble_mean.py` to ensemble the models and obtain final prediction results.

