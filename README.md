# RL-multiagent-communication

This is the GitHub repository for our course project, InSPIRe: Interpretable Sharing of Proactive Intention
in Reinforcement-learning, for 10-701 Introduction to Machine Learning at Carnegie Mellon University, Fall 2024 semester.

![](https://github.com/ABS510/RL-multiagent-communication/blob/main/Example%20Game%20Play.mov)

## Environment setup
We recommend creating a virtual environment for running this project. Please run the following commands to install the appropriate packages and install the Atari ROM. 

```
$ pip install -r requirements.txt
$ AutoROM --accept-license
```

## Running the training or evaluation experiments

To run a training or evaluation experiment, you should set up your own configuration file. Example training file is available at `config_sample.py` and an example evaluation file, with human-visible gameplay, is available at `config_eval.py`
```
$ python VolleyballPongEnv.py -c <path-to-your-config.py>
```

## Generating the plots from the log files

To generate the final plots from the log specified in the config's output directory:
```
$ python ParseLog.py -i <path-to-log-file.log> -o <path-to-output-directory> -m "optional string for plot titles"
```
