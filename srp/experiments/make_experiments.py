""" Generate modules for all combinations of settings we plan to test.

"""
from builtins import str
import os
from collections import OrderedDict
import itertools
import pandas as pd
import oyaml as yaml
from tqdm import tqdm
from srp.config import C


def make_experiments(options, dirname=C.INT_DATA, progress=True):
    """
    Generate grid-search parameters to explore a set of options.


    The idea is that you can write a single script that reads in a config file
    to do an experiments. The config files will be generated by this function.

    This script will not actually _do_ the experiments, it just generates the
    configuration files.

    :param options: An ordered dictionary; the first keys will vary fastest
                    in a grid search.
    :param dirname: A folder that will hold configuration settings for
                    each combination of options we explore.
    :param progress: whether to show a progress bar
    Example
    -------

    >>> options = yaml.load('''
    ... A:
    ...    - 1
    ...    - 2
    ...    - 3
    ... B:
    ...     - alpha
    ...     - beta
    ... C:
    ...     - orange
    ...     - yellow
    ... ''')
    >>> os.makedirs('data/test', exist_ok=True)
    >>> make_experiments(options, dirname='data/test', progress=False)
    >>> os.path.isfile('data/test/experiments.csv')
    True
    >>> os.path.isfile('data/test/options.yml')
    True

    The `experiments.csv` file has a header and a row with the settings for
    each trial.  The first option is changes fastest.
    >>> f = open('data/test/experiments.csv').readlines()
    >>> print(f[0].strip())
    ,A,B,C

    >>> print(f[1].strip())
    0,1,alpha,orange

    >>> print(f[2].strip())
    1,2,alpha,orange


    Files are created for each combination of options
    >>> os.path.isfile('data/test/experiments/00000/config.yml')
    True

    The experiment config files have al of the options as YAML data
    >>> c = yaml.load(open('data/test/experiments/00000/config.yml'))
    >>> c['A']
    1
    >>> c['B']
    'alpha'
    >>> c['C']
    'orange'
    """
    # Save the options
    with open(os.path.join(dirname, 'options.yml'), 'w') as f:
        yaml.dump(options, f, default_flow_style=False)

    # Save the master list of experiments
    combos = [reversed(combo) for combo in itertools.product(*reversed(list(options.values())))]
    experiments = pd.DataFrame(combos, columns=list(options.keys()))
    experiments.to_csv(os.path.join(dirname, 'experiments.csv'))

    # Make s folder and config file for each experiment
    if progress:
        combos = tqdm(experiments.iterrows(), "generating configs")
    else:
        combos = experiments.iterrows()

    for _, combo in combos:
        subdirname = '{:05}'.format(combo.name)

        rec = OrderedDict()
        rec['NUMBER'] = combo.name
        rec['NAME'] = '-'.join((subdirname, ) + tuple((str(x) for x in combo)))
        rec.update(combo)

        exp_dir = os.path.join(dirname, 'experiments', subdirname)
        os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, 'config.yml'), 'w') as f:
            yaml.dump(rec, f, default_flow_style=False)
