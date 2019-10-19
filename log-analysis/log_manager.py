"""
Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import ast
import json
import time
import os.path
import dateutil.parser

from decimal import *
from datetime import datetime

import boto3
import numpy as np
import pandas as pd

log_record = {
    'id': '', 'model_name': np.nan, 'log_type': '', 'world_name': '', 'cloned': np.nan,
    'fetched_at': '', 'first_event_timestamp': '', 'last_event_timestamp': '',
    'max_steering_angle': '', 'steering_angle_granularity': '', 'max_speed': '',
    'speed_granularity': '', 'batch_size': '', 'beta_entropy': '', 'discount_factor': '',
    'loss_type': '', 'learning_rate': '', 'num_epochs': '', 'num_episodes_between_training': '',
    'n_trials': np.nan, 'training_job_duration': np.nan, 's3_bucket': '', 's3_prefix': ''
}

log_record_dtypes = [object, object, object, object, object, datetime, datetime, datetime,
                     object, object, object, object, object, object, object, object, object,
                     object, object, int, object, object, object]

logs = {
    'types': {
        'sagemaker_bootstrap': 'training-job',
        'distributed_training.launch': 'simulation-job',
        'evaluation.launch': 'evaluation',
        'leaderboard': 'leaderboard',
    },
    'groups': {
        "sagemaker": '/aws/sagemaker/TrainingJobs',
        "robomaker": '/aws/robomaker/SimulationJobs',
        "leaderboard": '/aws/deepracer/leaderboard/SimulationJobs',
    },
    'paths': {
        'cloudwatch': 'logs/cloudwatch',
        'local_training': 'logs/local_training'
    },
    'db': {
        'cloudwatch': 'logs/db/cloudwatch_logs.csv',
        'local_training': 'logs/db/local_training_logs.csv',
    }
}


#
##
# MARK: Create csv files if not existed
##
if not os.path.isfile(logs['db']['cloudwatch']):
    pd.DataFrame([], columns=log_record.keys()) \
        .to_csv(logs['db']['cloudwatch'], index=False)

if not os.path.isfile(logs['db']['local_training']):
    pd.DataFrame([], columns=log_record.keys()) \
        .to_csv(logs['db']['local_training'], index=False)


##
# MARK: Methods for Fetching Logs from Cloudwatch
##
def fetch_all_logs_from_cloudwatch(log_ids=None, from_date=None, to_date=None, force=False):
    """
    It does the following:
    - Download all logs of all log groups from cloudwatch.
    - Collect data from each log.
    - Store collected data in cloudwatch_logs.csv.

    Parameters
    ----------
    from_date : datetime
        Don't include any logs before this datetime.
    to_date : datetime
        Don't include any logs after this datetime.
    force : bool, default False
       Whether to force download the log file if exists or not

    """

    # fetch_sagemaker_logs_from_cloudwatch(log_ids, from_date, to_date, force)
    fetch_robomaker_logs_from_cloudwatch(log_ids, from_date, to_date, force)
    fetch_leaderboard_logs_from_cloudwatch(log_ids, from_date, to_date, force)


# TODO - Implementation
def fetch_sagemaker_logs_from_cloudwatch(log_ids=None, from_date=None, to_date=None, force=False):
    pass


def fetch_robomaker_logs_from_cloudwatch(log_ids=None, from_date=None, to_date=None, force=False):
    """
    It does the following:
    - Download all logs of /aws/robomaker/SimulationJobs log group from cloudwatch.
    - Collect data from each log.
    - Store collected data in cloudwatch_logs.csv.

    Parameters
    ----------
    log_ids : list
        If exist, logs matches these log ids will only be downloaded
    from_date : datetime
        Don't include any logs before this datetime.
    to_date : datetime
        Don't include any logs after this datetime.
    force : bool, default False
       Whether to force download the log file if exists or not

    """

    db_file = logs['db']['cloudwatch']
    log_group = logs['groups']['robomaker']
    cloudwatch_log_path = logs['paths']['cloudwatch']

    if isinstance(log_ids, type(None)):
        fetched_logs = download_logs_from_cloudwatch_by_datetime(log_group,
                                                                 cloudwatch_log_path, from_date, to_date, force)
    else:
        fetched_logs = download_logs_from_cloudwatch(log_group, cloudwatch_log_path, log_ids, force)

    store_data_from_fetched_logs(db_file, log_group, fetched_logs)


def fetch_leaderboard_logs_from_cloudwatch(log_ids=None, from_date=None, to_date=None, force=False):
    """
    It does the following:
    - Download all logs of /aws/deepracer/leaderboard/SimulationJobs log group from cloudwatch.
    - Collect data from each log.
    - Store collected data in cloudwatch_logs.csv.

    Parameters
    ----------
    log_ids : list
        If exist, logs matches these log ids will only be downloaded
    from_date : datetime
        Don't include any logs before this datetime.
    to_date : datetime
        Don't include any logs after this datetime.
    force : bool, default False
       Whether to force download the log file if exists or not

    """
    pass

    csv = logs['db']['cloudwatch']
    log_group = logs['groups']['leaderboard']
    cloudwatch_log_path = logs['paths']['cloudwatch']

    if isinstance(log_ids, type(None)):
        fetched_logs = download_logs_from_cloudwatch_by_datetime(log_group,
                                                                 cloudwatch_log_path, from_date, to_date, force)
    else:
        fetched_logs = download_logs_from_cloudwatch(log_group, cloudwatch_log_path, log_ids, force)

    store_data_from_fetched_logs(csv, log_group, fetched_logs)


def download_logs_from_cloudwatch(log_group, log_path, log_ids=None, force=False):

    if not isinstance(log_ids, list):
        print_error("'log_ids' argument should be '{}' rather than '{}'".format(type([]), type(log_ids)))
        return []

    client = boto3.client('logs')

    fetched_files = []
    next_token = None

    while next_token is not 'theEnd':
        streams = describe_log_streams(client, log_group, next_token)

        next_token = streams.get('nextToken', 'theEnd')

        for stream in streams['logStreams']:
            fetched_log_id = stream['logStreamName'].split("/")[0]

            if fetched_log_id in log_ids:
                file_name = "%s/%s.log" % (log_path, fetched_log_id)

                if not os.path.isfile(file_name) or force:
                    download_log(file_name, stream_prefix=fetched_log_id, log_group=log_group, force=force)

                fetched_at = pd.to_datetime(round(time.time()), unit='s')
                fetched_files.append(
                    (file_name, fetched_log_id, fetched_at, stream['firstEventTimestamp'], stream['lastEventTimestamp']))

                log_ids.remove(fetched_log_id)
                if len(log_ids) == 0:
                    break

    if len(log_ids) > 0:
        print_warning("Log group '{}' doesn't have the following logs: {}".format(log_group, log_ids))

    return fetched_files


def download_logs_from_cloudwatch_by_datetime(log_group, log_path, from_date=None, to_date=None, force=False):
    """
    Download all cloudwatch logs that is of type evaluation.

    Parameters
    ----------
    log_group : str
        Cloudwatch log group, ex: /aws/sagemaker/TrainingJobs.
    log_path : str
        The path where cloudwatch logs are saved.
    from_date : datetime
        Don't include any logs before this datetime.
    to_date : datetime
        Don't include any logs after this datetime.
    force : bool, default False
       Whether to force download the log file if exists or not.

    """

    client = boto3.client('logs')

    lower_timestamp = iso_to_timestamp(from_date)
    upper_timestamp = iso_to_timestamp(to_date)

    fetched_files = []
    next_token = None

    while next_token is not 'theEnd':
        streams = describe_log_streams(client, log_group, next_token)

        next_token = streams.get('nextToken', 'theEnd')

        for stream in streams['logStreams']:
            stream_prefix = stream['logStreamName'].split("/")[0]

            if lower_timestamp and stream['lastEventTimestamp'] < lower_timestamp:
                return fetched_files  # we're done, next logs will be even older

            if upper_timestamp and stream['firstEventTimestamp'] > upper_timestamp:
                continue

            file_name = "%s/%s.log" % (log_path, stream_prefix)

            if not os.path.isfile(file_name) or force:
                download_log(file_name, stream_prefix=stream_prefix, log_group=log_group, force=force)

            fetched_at = pd.to_datetime(round(time.time()), unit='s')
            fetched_files.append(
                (file_name, stream_prefix, fetched_at, stream['firstEventTimestamp'], stream['lastEventTimestamp']))

    return fetched_files


def describe_log_streams(client, log_group, next_token):
    if next_token:
        streams = client.describe_log_streams(logGroupName=log_group, orderBy='LastEventTime',
                                              descending=True, nextToken=next_token)
    else:
        streams = client.describe_log_streams(logGroupName=log_group, orderBy='LastEventTime',
                                              descending=True)
    return streams


def download_log(fname, stream_name=None, stream_prefix=None,
                 log_group=None, start_time=None, end_time=None, force=False):
    if os.path.isfile(fname) and not force:
        print_warning('Log file exists, use force=True to download again')
        return

    if start_time is None:
        start_time = 1451490400000  # 2018
    if end_time is None:
        end_time = 2000000000000  # 2033 #arbitrary future date
    if log_group is None:
        log_group = "/aws/robomaker/SimulationJobs"

    with open(fname, 'w') as f:
        log_events = get_log_events(
            log_group=log_group,
            stream_name=stream_name,
            stream_prefix=stream_prefix,
            start_time=start_time,
            end_time=end_time
        )

        for event in log_events:
            f.write(event['message'].rstrip())
            f.write("\n")


def get_log_events(log_group, stream_name=None, stream_prefix=None, start_time=None, end_time=None):
    client = boto3.client('logs')
    if stream_name is None and stream_prefix is None:
        print("both stream name and prefix can't be None")
        return

    kwargs = {
        'logGroupName': log_group,
        'logStreamNames': [stream_name],
        'limit': 10000,
    }

    if stream_prefix:
        kwargs = {
            'logGroupName': log_group,
            'logStreamNamePrefix': stream_prefix,
            'limit': 10000,
        }

    kwargs['startTime'] = start_time
    kwargs['endTime'] = end_time

    while True:
        resp = client.filter_log_events(**kwargs)
        yield from resp['events']
        try:
            kwargs['nextToken'] = resp['nextToken']
        except KeyError:
            break


def store_data_from_fetched_logs(db_file, log_group, fetched_logs):
    """
    It does the following:
    - Download all logs of /aws/deepracer/leaderboard/SimulationJobs log group from cloudwatch.
    - Collect data from each log.
    - Store collected data in cloudwatch_logs.csv.

    Parameters
    ----------
    db_file : str
        Don't include any logs before this datetime.
    log_group: str
        Don't include any logs after this datetime.
    fetched_logs : list, default False
       Whether to force download the log file if exists or not

    """
    db = load_all_cloudwatch_logs(full=True)

    for log in fetched_logs:
        file_name = log[0]
        fetched_at = log[2]
        first_event_timestamp = log[3]
        last_event_timestamp = log[4]

        log_id = file_name.split('/')[-1].replace('.log', '')

        if len(db[db['id'] == log_id]) > 0:
            continue

        data = collect_data_from_log(file_name, log_group)

        if data['id'] == '' or data['log_type'] == '' \
                or data['world_name'] == '' or data['s3_bucket'] == '':
            print_warning("Data of log '{}' couldn't be collected, "
                  "maybe different log format, or a failed job.".format(log_id))
            continue

        data_df = pd.DataFrame([data.values()], columns=data.keys())
        data_df['fetched_at'] = fetched_at
        data_df['first_event_timestamp'] = pd.to_datetime(round(first_event_timestamp/1000), unit='s')
        data_df['last_event_timestamp'] = pd.to_datetime(round(last_event_timestamp/1000), unit='s')

        db = db.append(data_df, ignore_index=True, sort=False)

    # For evaluation logs, update model_name from any training-job or simulation-job matching s3_prefix
    logs_with_issue = db[db.model_name.isnull()]
    logs_without_issue = db[db.model_name.notnull()]

    for index, row in logs_with_issue.iterrows():
        matched_logs = logs_without_issue[(logs_without_issue['s3_prefix'] == row['s3_prefix'])]
        if len(matched_logs) > 0:
            db.loc[db.id == row.id, 'model_name'] = matched_logs['model_name'].iloc[0]

    save_cloudwatch_db(db)


##
# MARK: Methods for Loading Logs from cloudwatch_logs.csv
##
def load_all_cloudwatch_logs(full=False, some_columns=None):
    """
    Load logs from cloudwatch_logs.csv as pandas Dataframe.

    Parameters
    ----------
    full : bool, default False
        If True all columns will be loaded, by default some_columns will only be loaded
    some_columns: list
        List of columns that only needed to be displayed from the database

    """

    db_file = logs['db']['cloudwatch']

    if not some_columns:
        some_columns = ['id', 'model_name', 'log_type', 'world_name', 'fetched_at', 'first_event_timestamp']

    db = pd.read_csv(db_file, parse_dates=True).sort_values('fetched_at', ascending=False)
    return db if full else db[some_columns]


def load_cloudwatch_training_logs(full=False, some_columns=None):
    """
    Load training-job logs from cloudwatch_logs.csv as pandas Dataframe.

    Parameters
    ----------
    full : bool, default False
        If True all columns will be loaded, by default some_columns will only be loaded
    some_columns: list
        List of columns that only needed to be displayed from the database

    """

    if not some_columns:
        some_columns = ['id', 'model_name', 'log_type', 'world_name', 'fetched_at',
                        'training_job_duration', 'cloned', 'first_event_timestamp']

    df = load_all_cloudwatch_logs(full, some_columns)

    return df[df['log_type'] == 'training-job']


def load_cloudwatch_sim_logs(full=False, some_columns=None):
    """
    Load simulation-job logs from cloudwatch_logs.csv as pandas Dataframe.

    Parameters
    ----------
    full : bool, default False
        If True all columns will be loaded, by default some_columns will only be loaded
    some_columns: list
        List of columns that only needed to be displayed from the database

    """

    if not some_columns:
        some_columns = ['id', 'model_name', 'log_type', 'world_name', 'fetched_at', 'first_event_timestamp']

    df = load_all_cloudwatch_logs(full, some_columns)

    return df[df['log_type'] == logs['types']['distributed_training.launch']]


def load_cloudwatch_eval_logs(full=False, some_columns=None):
    """
    Load evaluation logs from cloudwatch_logs.csv as pandas Dataframe.

    Parameters
    ----------
    full : bool, default False
        If True all columns will be loaded, by default some_columns will only be loaded
    some_columns: list
        List of columns that only needed to be displayed from the database

    """

    if not some_columns:
        some_columns = ['id', 'model_name', 'log_type', 'world_name', 'n_trials', 'fetched_at', 'first_event_timestamp']

    df = load_all_cloudwatch_logs(full, some_columns)

    return df[df['log_type'] == logs['types']['evaluation.launch']].copy()


def load_cloudwatch_leaderboard_logs(full=False, some_columns=None):
    """
    Load leaderboard logs from cloudwatch_logs.csv as pandas Dataframe.

    Parameters
    ----------
    full : bool, default False
        If True all columns will be loaded, by default some_columns will only be loaded
    some_columns: list
        List of columns that only needed to be displayed from the database

    """

    if not some_columns:
        some_columns = ['id', 'model_name', 'log_type', 'world_name', 'n_trials', 'fetched_at', 'first_event_timestamp']

    df = load_all_cloudwatch_logs(full, some_columns)

    return df[df['log_type'] == logs['types']['leaderboard']].copy()


##
# MARK: Methods for Generating Analysis Notebooks
##
def gen_analysis_notebook_for_cloudwatch_logs(log_ids, force=False):
    if not isinstance(log_ids, list):
        print_error("log_ids argument should be list not ''".format(type(log_ids)))
        return

    for log_id in log_ids:
        log = find_cloudwatch_log_record(log_id)

        if len(log) == 0:
            print_warning("Couldn't find log entry with id '{}'".format(log_id))
            continue

        log_type = log['log_type'].iloc[0]

        if log_type == logs['types']['distributed_training.launch']:
            pass  # TODO - Implement

        elif log_type == logs['types']['evaluation.launch']:
            gen_eval_analysis_notebook_for_cloudwatch_logs(log_id, force)

        elif log_type == logs['types']['leaderboard']:
            gen_eval_analysis_notebook_for_cloudwatch_logs(log_id, force, for_submission=True)


def gen_eval_analysis_notebook_for_cloudwatch_logs(log_id, force=False, for_submission=False):
    log_analysis_notebook_template = 'templates/Log Analysis for Evaluation or Submission - Template.ipynb'

    if for_submission:
        log_analysis_notebook = 'Log Analysis for Submission - {}.ipynb'.format(log_id)
    else:
        log_analysis_notebook = 'Log Analysis for Evaluation - {}.ipynb'.format(log_id)

    if not force and os.path.exists(log_analysis_notebook):
        print_warning('Notebook exists, use force=True to overwrite it')
        return

    log = find_cloudwatch_log_record(log_id)
    log_file = '{}/{}.log'.format(logs['paths']['cloudwatch'], log_id)

    with open(log_analysis_notebook_template, 'r') as notebook:
        notebook_as_text = json.load(notebook)
        cells = notebook_as_text['cells']

        title_and_log_details_cell = cells[0]['source']
        title_and_log_details_cell[0] = title_and_log_details_cell[0].format('Submission' if for_submission else 'Evaluation')

        log_details_vars_cell = cells[1]['source']
        log_details_vars_cell[0] = log_details_vars_cell[0].format(log_id)
        log_details_vars_cell[1] = log_details_vars_cell[1].format(log['log_type'].iloc[0])
        log_details_vars_cell[2] = log_details_vars_cell[2].format(log['model_name'].iloc[0])
        log_details_vars_cell[3] = log_details_vars_cell[3].format(log['world_name'].iloc[0])
        log_details_vars_cell[4] = log_details_vars_cell[4].format(int(log['n_trials'].iloc[0]))
        log_details_vars_cell[5] = log_details_vars_cell[5].format(log_file)

        load_waypoints_title = cells[4]['source']
        load_waypoints_title[0] = load_waypoints_title[0].format(log['world_name'].iloc[0])

    with open(log_analysis_notebook, 'w') as notebook:
        json.dump(notebook_as_text, notebook)

    command = "jupyter nbconvert --execute --inplace '{}' 2>&1".format(log_analysis_notebook)
    print(os.popen(command).read())


# def gen_leaderboard_analysis_notebook_for_cloudwatch_logs(log_id, force=False):
#     analysis_notebook_template = 'Log Analysis for Evaluation or Submission - Template.ipynb'
#     submission_analysis_notebook = 'Log Analysis for Evaluation - {}.ipynb'.format(log_id)
#
#     if not force and os.path.exists(submission_analysis_notebook):
#         print_warning('Notebook exists, use force=True to overwrite it')
#         return
#
#     log = find_cloudwatch_log_record(log_id)
#     log_file = '{}/{}.log'.format(logs['paths']['cloudwatch'], log_id)
#
#     with open(analysis_notebook_template, 'r') as notebook:
#         notebook_as_text = json.load(notebook)
#         cells = notebook_as_text['cells']
#
#         title_and_log_details_cell = cells[0]['source']
#         title_and_log_details_cell[0] = title_and_log_details_cell[0].format('Submission')
#
#         log_details_vars_cell = cells[1]['source']
#         log_details_vars_cell[0] = log_details_vars_cell[0].format(log_id)
#         log_details_vars_cell[1] = log_details_vars_cell[1].format(log['log_type'].iloc[0])
#         log_details_vars_cell[2] = log_details_vars_cell[2].format(log['model_name'].iloc[0])
#         log_details_vars_cell[3] = log_details_vars_cell[3].format(log['world_name'].iloc[0])
#         log_details_vars_cell[4] = log_details_vars_cell[4].format(int(log['n_trials'].iloc[0]))
#         log_details_vars_cell[5] = log_details_vars_cell[5].format(log_file)
#
#         load_waypoints_title = cells[4]['source']
#         load_waypoints_title[0] = load_waypoints_title[0].format(log['world_name'].iloc[0])
#
#     with open(submission_analysis_notebook, 'w') as notebook:
#         json.dump(notebook_as_text, notebook)
#
#     command = "jupyter nbconvert --execute --inplace '{}' 2>&1".format(submission_analysis_notebook)
#     print(os.popen(command).read())


def delete_cloudwatch_logs(log_ids):
    """
    Delete log entries from cloudwatch_logs.csv.
    NOTE: This won't delete logs files themselves, only their entries from database.

    Parameters
    ----------
    log_ids : list
        List of logs ids to delete.

    """

    if not isinstance(log_ids, list):
        print_error("log_ids should be list not ''".format(type(log_ids)))

    df = load_all_cloudwatch_logs(full=True)

    print_info('Deleting the following log entries: {}'.format(', '.join(log_ids)))

    while True:
        print('Confirm deletion (y/N): ', end=" ")
        answer = input()
        if answer.lower().strip() in ['y', 'yes']:
            df = df[~df['id'].isin(log_ids)]

            save_cloudwatch_db(df)

            print_info('Logs entries deleted successfully!')
            break

        elif answer.lower().strip() in ['', 'n', 'no']:
            print_info('Operation canceled!')
            break
        else:
            print("Wrong input, please specify 'y' or 'n'")


def extract_required_lines(file_name, log_group):
    # TODO - Implement extract_lines_from_sagemaker_log function
    if log_group == logs['groups']['sagemaker']:
        pass

    elif log_group == logs['groups']['robomaker'] \
            or log_group == logs['groups']['leaderboard']:
        return extract_required_lines_from_robomaker_log(file_name)

    else:
        print_error('Invalid log group "{}"'.format(log_group))


# TODO - Implement
def extract_required_lines_from_sagemaker_log(file_name):
    pass


def extract_required_lines_from_robomaker_log(file_name):
    required_lines = []

    with open(file_name) as file:
        for line in file:
            line = line.strip()

            if line.startswith('robomaker job description: ') \
                    or line.startswith('Loaded action space from file: ') \
                    or '"lr": ' in line \
                    or '"loss_type": ' in line \
                    or '"num_epochs": ' in line \
                    or '"batch_size": ' in line \
                    or '"beta_entropy": ' in line \
                    or '"discount_factor": ' in line \
                    or '"num_episodes_between_training": ' in line:
                required_lines.append(line)

            number_of_required_lines = 9  # Please edit this number of you are searching for more lines
            if len(required_lines) == number_of_required_lines:
                break

    return required_lines


def collect_data_from_log(file_name, log_group):
    # TODO - Implement collect_data_from_sagemaker_log
    if log_group == logs['groups']['sagemaker']:
        return collect_data_from_sagemaker_log(file_name)

    elif log_group == logs['groups']['robomaker']:
        return collect_data_from_robomaker_log(file_name)

    elif log_group == logs['groups']['leaderboard']:
        return collect_data_from_leaderboard_log(file_name)

    else:
        print_error('Invalid log group "{}"'.format(log_group))


# TODO - Implement
def collect_data_from_sagemaker_log(file_name):
    pass


def collect_data_from_robomaker_log(file_name):
    log_group = logs['groups']['robomaker']

    data = log_record.copy()

    required_lines = extract_required_lines(file_name, log_group)

    # TODO - We need to implement what to do when the robomaker log is for failed job, shall we keep it?
    # if len(required_lines) == 0:
    #     required_lines = try_again_extract_required_lines();

    for line in required_lines:

        if line.startswith('Loaded action space from file: '):
            action_space = action_space_line_to_array(line)
            collect_data_from_action_space(data, action_space)

        elif line.startswith("robomaker job description: "):
            job_description = robomaker_job_description_line_to_json(line)

            launch_config = job_description['simulationApplications'][0]['launchConfig']
            data['log_type'] = logs['types'][launch_config['launchFile']]

            collect_data_from_robomaker_job_description(data, job_description)

        else:
            collect_hyperparameters_from_robomaker_log_line(data, line)

    return data


def collect_data_from_leaderboard_log(file_name):
    log_group = logs['groups']['leaderboard']

    data = log_record.copy()

    required_lines = extract_required_lines(file_name, log_group)

    # TODO - We need to implement what to do when the robomaker log is for failed job, shall we keep it?
    # if len(required_lines) == 0:
    #     required_lines = try_again_extract_required_lines();

    for line in required_lines:

        if line.startswith('Loaded action space from file: '):
            action_space = action_space_line_to_array(line)
            collect_data_from_action_space(data, action_space)

        elif line.startswith("robomaker job description: "):
            job_description = robomaker_job_description_line_to_json(line)

            data['log_type'] = logs['types']['leaderboard']

            collect_data_from_robomaker_job_description(data, job_description)

        else:
            collect_hyperparameters_from_robomaker_log_line(data, line)

    return data


def collect_data_from_action_space(data, action_space):
    steering_angle_set = set([])
    speed_set = set([])

    for item in action_space:
        steering_angle_set.add(item['steering_angle'])
        speed_set.add(item['speed'])

    data['max_steering_angle'] = max(steering_angle_set)
    data['steering_angle_granularity'] = len(steering_angle_set)
    data['max_speed'] = max(speed_set)
    data['speed_granularity'] = len(speed_set)


def collect_data_from_robomaker_job_description(data, job_description):
    arn = job_description['arn'].split('/')[-1]
    environment_variables = job_description['simulationApplications'][0]['launchConfig']['environmentVariables']

    data['id'] = arn
    data['world_name'] = environment_variables['WORLD_NAME']

    if data['log_type'] == logs['types']['distributed_training.launch']:
        data['model_name'] = extracted_model_name_from_job_description(job_description)
        data['s3_bucket'] = environment_variables['SAGEMAKER_SHARED_S3_BUCKET']
        data['s3_prefix'] = environment_variables['SAGEMAKER_SHARED_S3_PREFIX']

    elif data['log_type'] == logs['types']['evaluation.launch']:
        data['n_trials'] = int(environment_variables['NUMBER_OF_TRIALS'])
        data['s3_bucket'] = environment_variables['METRICS_S3_BUCKET']
        data['s3_prefix'] = environment_variables['MODEL_S3_PREFIX']

    elif data['log_type'] == logs['types']['leaderboard']:
        data['n_trials'] = int(environment_variables['NUMBER_OF_TRIALS'])
        data['s3_bucket'] = environment_variables['METRICS_S3_BUCKET']
        data['s3_prefix'] = environment_variables['MODEL_S3_PREFIX']


def collect_hyperparameters_from_robomaker_log_line(data, line):
    if '"batch_size"' in line:
        data['batch_size'] = int(line.split(':')[-1].replace(',', '').strip())

    elif '"beta_entropy"' in line:
        data['beta_entropy'] = float(line.split(':')[-1].replace(',', '').strip())

    elif '"discount_factor"' in line:
        data['discount_factor'] = float(line.split(':')[-1].replace(',', '').strip())

    elif '"loss_type"' in line:
        data['loss_type'] = line.split(':')[-1].replace(',', '').replace('"', '').strip()

    elif '"lr"' in line:
        data['learning_rate'] = float(line.split(':')[-1].replace(',', '').strip())

    elif '"num_epochs"' in line:
        data['num_epochs'] = int(line.split(':')[-1].replace(',', '').strip())

    elif '"num_episodes_between_training"' in line:
        data['num_episodes_between_training'] = int(line.split(':')[-1].replace(',', '').strip())


def extracted_model_name_from_job_description(job_description):
    environment_variables = job_description['simulationApplications'][0]['launchConfig'][
        'environmentVariables']
    s3_key = environment_variables['REWARD_FILE_S3_KEY']

    if s3_key.startswith('reward-functions'):
        index = s3_key.find('/')
        s3_key = s3_key[index + 1:]
        index = s3_key.rfind('/')
        return s3_key[:index]


def action_space_line_to_array(line):
    return ast.literal_eval(
        line.replace('Loaded action space from file: ', ''))


def robomaker_job_description_line_to_json(line):
    line = remove_datetime_fields(line)
    line = line \
        .replace('robomaker job description: ', '') \
        .replace('\'', '\"') \
        .replace('True', 'true') \
        .replace('False', 'false')

    return json.loads(line)


def remove_datetime_fields(line):
    datetime_fields = ['lastUpdatedAt', 'lastStartedAt']

    for field_name in datetime_fields:
        upper_index = line.find("'{}': ".format(field_name))

        if upper_index != -1:

            lower_index = upper_index + 10
            while True:
                text = line[lower_index-10: lower_index]
                if text == 'tzlocal())':
                    break
                lower_index += 1

            lower_index = (lower_index + 2) if line[lower_index+1] == ',' else (lower_index + 1)

            line = line[:upper_index] + line[lower_index:]

    return line


##
# MARK: Methods for Describing Action Space & Hyper-Parameters
##
def describe_action_space(log_id):
    log = find_cloudwatch_log_record(log_id)

    max_steering_angle = log['max_steering_angle'].iloc[0]
    steering_angle_granularity = log['steering_angle_granularity'].iloc[0]
    max_speed = log['max_speed'].iloc[0]
    speed_granularity = log['speed_granularity'].iloc[0]

    steering_angle_list = np.linspace(-max_steering_angle, max_steering_angle, num=steering_angle_granularity).tolist() * speed_granularity
    speed_list = [(max_speed - i * speed_granularity) for i in range(speed_granularity-1, -1, -1)] * steering_angle_granularity

    index = ['steering_angle (degrees)', 'speed (m/s)']
    header = list(range(0, (speed_granularity * steering_angle_granularity)))

    return pd.DataFrame([steering_angle_list, speed_list], columns=header, index=index)


def describe_hyper_parameters(log_id):
    log = find_cloudwatch_log_record(log_id)

    hyper_parameters = {
        'batch_size': log['batch_size'].iloc[0],
        'beta_entropy': log['beta_entropy'].iloc[0],
        'discount_factor': log['discount_factor'].iloc[0],
        'loss_type': log['loss_type'].iloc[0],
        'learning_rate': log['learning_rate'].iloc[0],
        'num_epochs': log['num_epochs'].iloc[0],
        'num_episodes_between_training': log['num_episodes_between_training'].iloc[0],
    }

    return pd.DataFrame(hyper_parameters.values(), index=hyper_parameters.keys(), columns=['Value'])


##
# MARK: Methods Used for Analysis Notebooks
##
def load_data(fname):
    from os.path import isfile
    data = []

    i = 1

    while isfile('%s.%s' % (fname, i)):
        load_file('%s.%s' % (fname, i), data)
        i += 1

    load_file(fname, data)

    if i > 1:
        print("Loaded %s log files (logs rolled over)" % i)

    return data


def load_file(log_file, data):
    with open(log_file, 'r') as f:
        for line in f.readlines():
            if "SIM_TRACE_LOG" in line:
                parts = line.split("SIM_TRACE_LOG:")[1].split('\t')[0].split(",")
                data.append(",".join(parts))


def convert_to_pandas(data, episodes_per_iteration=20):
    """
    stdout_ = 'SIM_TRACE_LOG:%d,%d,%.4f,%.4f,%.4f,%.2f,%.2f,%d,%.4f,%s,%s,%.4f,%d,%.2f,%s\n' % (
            self.episodes, self.steps, model_location[0], model_location[1], model_heading,
            self.steering_angle,
            self.speed,
            self.action_taken,
            self.reward,
            self.done,
            all_wheels_on_track,
            current_progress,
            closest_waypoint_index,
            self.track_length,
            time.time())
        print(stdout_)
    """

    df_list = list()

    # ignore the first two dummy values that coach throws at the start.
    for d in data[2:]:
        parts = d.rstrip().split(",")
        episode = int(parts[0])
        steps = int(parts[1])
        x = 100 * float(parts[2])
        y = 100 * float(parts[3])
        yaw = float(parts[4])
        steer = float(parts[5])
        throttle = float(parts[6])
        action = float(parts[7])
        reward = float(parts[8])
        done = 0 if 'False' in parts[9] else 1
        all_wheels_on_track = parts[10]
        progress = float(parts[11])
        closest_waypoint = int(parts[12])
        track_len = float(parts[13])
        tstamp = Decimal(parts[14])

        iteration = int(episode / episodes_per_iteration) + 1
        df_list.append((iteration, episode, steps, x, y, yaw, steer, throttle,
                        action, reward, done, all_wheels_on_track, progress,
                        closest_waypoint, track_len, tstamp))

    header = ['iteration', 'episode', 'steps', 'x', 'y', 'yaw', 'steer',
              'throttle', 'action', 'reward', 'done', 'on_track', 'progress',
              'closest_waypoint', 'track_len', 'timestamp']

    df = pd.DataFrame(df_list, columns=header)
    return df


def load_eval_data(log_file):
    eval_data = load_data(log_file)
    return convert_to_pandas(eval_data)


def load_eval_log(log_id):
    log_file = '{}/{}.log'.format(logs['paths']['cloudwatch'], log_id)

    full_dataframe = None

    eval_data = load_data(log_file)
    dataframe = convert_to_pandas(eval_data)
    dataframe['stream'] = log_id

    if full_dataframe is not None:
        full_dataframe = full_dataframe.append(dataframe)
    else:
        full_dataframe = dataframe

    return full_dataframe.sort_values(
        ['stream', 'episode', 'steps']).reset_index()


##
# MARK: Methods used as kind of helper/util methods
##
def iso_to_timestamp(iso_date):
    return dateutil.parser.parse(iso_date).timestamp() * 1000 if iso_date else None


def validate_boto_version():
    required_boto3_version = '1.9.133'.split('.')
    installed_boto3_version = boto3.__version__.split('.')

    for i in range(len(required_boto3_version)):
        if int(installed_boto3_version[i]) < int(required_boto3_version[i]):
            print_error("Installed boto version '{}' must be >= '{}'"
                .format('.'.join(installed_boto3_version), '.'.join(required_boto3_version)))
            return

    print_info("Installed boto version meets the minimum requirements, '{}' >= '{}'"
          .format('.'.join(installed_boto3_version), '.'.join(required_boto3_version)))


def find_cloudwatch_log_record(log_id):
    db = pd.read_csv(logs['db']['cloudwatch'])
    return db[db['id'] == log_id]


def save_cloudwatch_db(db):
    cloudwatch_db = logs['db']['cloudwatch']
    db.to_csv(cloudwatch_db, index=False)


def print_info(message):
    print("INFO: {}".format(message))


def print_warning(message):
    print("WARNING: {}".format(message))


def print_error(message):
    print("ERROR: {}".format(message))
