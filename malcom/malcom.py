#!/usr/bin/python3
import logging
import argparse

import malcom.experiments as experiments


def init_logger(log_level_str):
    if log_level_str == 'INFO':
        log_level = logging.INFO
    elif log_level_str == 'DEBUG':
        log_level = logging.DEBUG
    elif log_level_str == 'WARN':
        log_level = logging.WARN
    elif log_level_str == 'ERROR':
        log_level = logging.ERROR

    logger = logging.getLogger('')
    logger.setLevel(log_level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    f = '~%(levelname)s~ %(filename)s:%(funcName)s:%(lineno)s --> %(message)s'
    formatter = logging.Formatter(f)
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)


def init_parser():
    parser = argparse.ArgumentParser(
        description='Malcom: Predicting things',
        formatter_class=argparse.MetavarTypeHelpFormatter
    )
    parser.add_argument('--log_level', '-l', type=str, default='INFO', required=False)
    parser.add_argument('--experiment', '-e', type=str,
                        help='experiment configuration file', required=True)
    # parser.add_argument('--db', type=str, help='db name', required=False)
    return parser


# def plot_tpch10_select_error():
#     pth = "./docs/figs/tpch10/"
#     d = 'tpch10'
#     for i in set(range(1, 23)) - set([10, 13]):
#         q = "0{}".format(i) if i < 10 else str(i)
#         o = "./docs/figs/tpch10/tpch10_sel{}_error.pdf".format(q)
#         experiments.plot_select_error_air(d, q, ntrain=200, path=pth, output=o)
#
#
# def plot_tpch10_mem_error():
#     pth = "./docs/figs/tpch10/"
#     d = 'tpch10'
#     for i in set(range(1, 23)) - set([10, 13]):
#         q = "0{}".format(i) if i < 10 else str(i)
#         o = "./docs/figs/tpch10/tpch10_q{}_memerror.pdf".format(q)
#         experiments.plot_mem_error_air(d, q, ntrain=200, path=pth, output=o)
#
#
# def plot_airtraffic_select_error():
#     f = "./docs/figs/airtraffic/airtraffic_sel{}_error.pdf"
#     for q in ['04', '09', '10', '11', '15.1', '19.1']:
#         o = f.format(q.replace('.', '_'))
#         experiments.plot_select_error_air('airtraffic', q, trainq=q, output=o)
#
#
# def plot_airtraffic_mem_error():
#     f = "./docs/figs/airtraffic/airtraffic_q{}_memerror.pdf"
#     for q in ['04', '09', '10', '11', '15.1', '19.1']:
#         testq = q
#         out = f.format(q.replace('.', '_'))
#         experiments.plot_mem_error_air('airtraffic', testq, trainq=q, output=out)
#
#

experiment_dispatcher = {
    'leave one out': experiments.leave_one_out,
    'actual memory': experiments.plot_actual_memory
}


def default_function(definition):
    print('Unknown experiment type. Valid experiments are:')
    for k in experiment_dispatcher:
        print("  ", k)


def main():
    parser = init_parser()
    args = parser.parse_args()
    init_logger(args.log_level)
    experiment_definition = experiments.parse_experiment_definition(args.experiment)

    func = experiment_dispatcher.get(experiment_definition['experiment'], default_function)
    func(experiment_definition)


if __name__ == '__main__':
    main()

    # experiments.predict_max_mem_tpch10()
    # experiments.plot_mem_error_air('airtraffic','11',path="./")
    # experiments.analyze_mem_error_air('airtraffic',11)

    # experiments.plot_mem_error_air('airtraffic','09',path="./", output="airtraffic_09.2_memerror.pdf")
    # experiments.analyze_mem_error_air("tpch10",'16',ntrain=200, step=200)
    # experiments.analyze_mem_error_air("airtraffic",'09',ntrain=1000, step=500)
    # experiments.plot_memerror_tpch10(path="./docs/figs/tpch10/")
    # experiments.analyze_select_error_air('tpch10',17, ntrain=200, step=100)
    # experiments.analyze_select_error_air('tpch10',"18", ntrain=200, step=100)
    # plot_tpch10_select_error()
    # plot_airtraffic_select_error()
    # plot_airtraffic_mem_error()
    # plot_tpch10_mem_error()
    # experiments.plot_allmem_tpch10()
