#!/usr/bin/env python3
# Python 3.9 or higher required to run this program.

# pylint: disable=missing-docstring

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import List, NamedTuple

import pandas as pd
import numpy as np


logger = logging.getLogger('yaroslavl')


class ProgramOptions(NamedTuple):
    input_path: List[Path]
    output_path: Path
    tresholds: List[float]
    aggregate_function: str
    blend_targets: bool
    weights: List[List[float]]


TARGETS = [
    'Артериальная гипертензия',
    'ОНМК',
    'Стенокардия, ИБС, инфаркт миокарда',
    'Сердечная недостаточность',
    'Прочие заболевания сердца',
]

TARGETS_CORR = {
    'Артериальная гипертензия': [1, 0, 0, 0, 0],
    'ОНМК': [0, 1, 0, 0, 0],
    'Стенокардия, ИБС, инфаркт миокарда': [0, 0, 1, 0, 0],
    'Сердечная недостаточность': [0, 0, 0, 1, 0.5],
    'Прочие заболевания сердца': [0, 0, 0, 0.5, 1],
    }


def create_submission(options: ProgramOptions):
    logger.info('Read predicts...')
    data = [pd.read_csv(path) for path in options.input_path]
    predicts = []
    for target_num, target_name in enumerate(TARGETS):
        sum_predict = None
        sum_weight = 0.0
        for df_num, df in enumerate(data):
            predicts_per_target = (len(df.columns) - 1) // len(TARGETS)
            target_columns = [f'{target_name} - {i}' for i in range(predicts_per_target)]
            if options.aggregate_function == 'mean':
                predict = df[target_columns].mean(axis=1)
            elif options.aggregate_function == 'max':
                predict = df[target_columns].max(axis=1)

            if options.weights:
                weight = options.weights[df_num][target_num]
                predict *= weight
                sum_weight += weight
            else:
                sum_weight += 1

            if sum_predict is None:
                sum_predict = predict
            else:
                sum_predict += predict

        predicts.append(sum_predict / sum_weight)

    submission = data[0][['ID']].copy()
    for target_name, predict, treshold in zip(TARGETS, predicts, options.tresholds):
        if options.blend_targets:
            sum_predict = None
            for i in range(len(TARGETS)):
                p = predicts[i] * TARGETS_CORR[target_name][i]
                if sum_predict is None:
                    sum_predict = p
                else:
                    sum_predict += p
            result = (sum_predict / sum(TARGETS_CORR[target_name])) >= treshold
        else:
            result = predict >= treshold
        logger.info(f'{target_name}: {result.sum()}, {len(result)}, {result.sum() / len(result)}')
        submission[target_name] = result.astype(int)
        
    logger.info('Create submission...')
    submission.to_csv(options.output_path, index=False, encoding='utf-8', line_terminator='\n')


def configure_logging():
    formatter = logging.Formatter('%(asctime)s [%(levelname)5s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)

    logger.setLevel(1) # min level


def parse_command_line_options() -> ProgramOptions:
    parser = argparse.ArgumentParser(
        description='Create submission for the Yaroslavl regional contest (https://lk.hacks-ai.ru/758240/champ).',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i,--input-path', dest='input_path', action='append', type=Path,
                        help='path to the input csv-file that contains predicts')

    parser.add_argument('-o,--output-path', dest='output_path', type=Path, required=True,
                        help='path to the output csv-file')

    parser.add_argument('--tresholds', dest='tresholds', type=str, required=True,
                        help='predictions tresholds for all targets')

    parser.add_argument('--aggregate-function', dest='aggregate_function', type=str,
                        choices=['mean', 'max'],
                        help='aggregate function for predicts (default: %(default)s)')

    parser.add_argument('--blend-targets', dest='blend_targets', action='store_true',
                        help='blend targets using correlation matrix')

    parser.add_argument('-w,--weights', dest='weights', action='append', type=str,
                        help='weights for all targets')

    parser.set_defaults(aggregate_function='mean')

    args = parser.parse_args()

    for path in args.input_path:
        if not path.exists():
            parser.error(f'argument --input-path: path "{path}" not found')

    args.tresholds = list(map(float, args.tresholds.split(';')))
    if len(args.tresholds) != len(TARGETS):
        parser.error(f'argument --tresholds: invalid tresholds count {args.tresholds}')

    if args.weights:
        if len(args.weights) != len(args.input_path):
            parser.error(f'argument --weights: missed argument')
        weights_list = []
        for weights in args.weights:
            weights = list(map(float, weights.split(';')))
            if len(weights) != len(TARGETS):
                parser.error(f'argument --weights: invalid weights count {weights}')
            weights_list.append(weights)
        args.weights = weights_list

    return ProgramOptions(**vars(args))


def main():
    options = parse_command_line_options()

    configure_logging()

    start_time = time.perf_counter()
    create_submission(options)
    logger.info(f'Total time: {time.perf_counter() - start_time} sec')


if __name__ == '__main__':
    main()
