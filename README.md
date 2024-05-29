# SPI

This repository contains the code for the SPI (safe policy improvement) exercise

## Getting started

After cloning this repository:

1. create a virtualenv and activate it
```bash
cd univr_offline_rl/
python3 -m venv .venv
source .venv/bin/activate
```
2. install the dependencies
```bash
pip install -r requirements.txt
```

## Taxi experiments
 
This section shows how the codebase can be used to run new experiments.

### Training a behavior policy:

The `training_baseline_policy.py` can be used to generate a behavior policy.
```bash
python training_baseline_policy.py --env_id "Taxi-v3"  --training_episodes 5000 --decaying_rate_qlearning 0.002 --beta 0.6
```
```text
Q-Learning:  16%|█▊         | 842/5250 [00:00<00:01, 3098.25it/s, ret=8.5]
```


Use the following command for further options.
```bash
python training_baseline_policy.py --help
```


### Run SPI experiments

The following command runs the safe policy improvement experiments with seeds `1-2` using `1` process and the behavior policy from the file `data/Taxi-v3.pkl`.

```bash
python main.py -c 1 -s $(seq -s \  1 2) -p data/Taxi-v3.pkl
```

```text

```

Use the following command for further options.

```bash
python main.py --help
```


### Visualize the results

```bash
cd plotting/
python plot.py Taxi-v3 --show
```

