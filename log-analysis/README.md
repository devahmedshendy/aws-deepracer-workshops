# Log Analysis
This folder contains tools making it simple to read information from training and evaluation logs.

The main part of this project are [Jupyter](https://jupyter.org/) notebooks - think of it as a text editor document enriched with executable code.

## What you need to know to use this

If you just want to have a look, GitHub has a viewer for the notebooks, just click into them and enjoy.

For working with the notebooks you need to be familiar with Python code, but the whole process is reasonably simple. Getting to know pandas and matplotlib will help you evolve from the solutions provided to your own bespoke analysis toolkit.

Tinkering and trying things out is highly desirable. Please share your ideas

## About Added Features
- Data about deepracer logs will be stored in csv file as follows:
    - `logs/db/cloudwatch_logs.csv` for cloudwatch logs.  
    - `logs/db/local_training_logs.csv` for local_training training logs.  

- Log files will be stored in the following paths:
    - `logs/cloudwatch` for cloudwatch logs.
    - `logs/local_training` for local_training training logs.

- CSV Database Columns:
    - id
    - model_name
    - log_type
    - world_name
    - cloned (not-implemented-yet)
    - fetched_at
    - first_event_timestamp
    - last_event_timestamp
    - max_steering_angle
    - steering_angle_granularity
    - max_speed
    - speed_granularity
    - batch_size
    - beta_entropy
    - discount_factor
    - loss_type
    - learning_rate
    - num_epochs
    - num_episodes_between_training
    - n_trials
    - training_job_duration (not-implemented-yet)
    - s3_bucket
    - s3_prefix

- All logs whether they are from cloudwatch or local_training training will be saved in format `<log-type>-<log_id>.log`.  
  example `logs/cloudwatch/sim-1j6vm0shdd15.log`
  
- Logs types:
  - `training-job` - sagemaker log of training job
  - `simulation-job` - robomaker log of training job
  - `evaluation` - robomaker log of training evaluation
  - `leaderboard` - romomaker log of leaderboard submission
  
- **Notes**
  - Logs will always be read from database sorted in descending.
  - `evaluation` and `leaderboard` logs does not contain the model name, but `training-job` and `simulation-job` does have, so I look for `simulation-job` with same s3_prefix and use its model name. 
  - Some logs are not valid anymore because of many reasons, like there was error that stopped the training in the beginning, so how we will deal with them .. currently I ignore them, but their log file is already downloaded.
  - When a file is already existed but downloading it stopped or failed for some reason, how can we detect this? This needs to be implemented.
  - I did not make any verbose output for most of the functions, once features approved, we can implement these for better experience.
  - Track/World names in AWS DeepRacer console does not match the names under `log-analysis/tracks/` directory.  
    I do recommend AWS to set a standard names for training tracks, and submission tracks for future needs. 

## Notebooks

There are currently following notebooks:
* `DeepRacer Log Analysis.ipynb` - original notebook provided by the AWS folks (it has things not used in notebooks listed below)
* `Training_analysis.ipynb` - built on top of the first one with some things removed and many added, prepared to monitor the training progress
* `Evaluation_analysis.ipynb` - built on top of the first one, prepared to analyse evaluation data

## Running the notebooks

I recommend setting up a venv for this:
```
python3 -m venv venv
```
(I recommend folder venv as I have already added it to .gitignore)
Then activate:
```
source venv/bin/activate
```
Then install dependencies:
```
pip install shapely matplotlib pandas sklearn boto3 awscli jupyter
```
Then run
```
jupyter notebook
```
From the opened page you can select a notebook to work with.

## Useful hints
* logs and reward folders have been configured to be ignored by git. This is so that you don't accidentally submit your reward functions or other useful info. Just make sure you secure it somehow yourself.
* have a look at new_reward function usage in the notebooks. It lets you try and evaluate what the reward would look like for a different reward function.

## What can I contribute?

There is a number of opportunities for improvement:
* Report issues/feature requests
* Fix things
* Improve descriptions
* Provide more resources
* Add analysis bits to notebooks
* Complete the `logs_to_params` method in log_analysis to improve the logs replay for a different reward
* Fill in track data used in breakdown in `Training_analysis.ipynb`
* Make the notebooks work with more tracks
