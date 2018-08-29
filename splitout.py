#!/usr/bin/env python3

import argparse
import pickle
import os
import sys

import lz4.frame as lz4frame

from malcom.utils import Utils
from malcom.stats import ColumnStatsD
from malcom.mal_dict import MalDictionary

DESCRIPTION = """
This script reads the specified trace file and splits its contents by query tag.
For every tag a compressed pickled maldict is written to a file whose name is
derived from the output_files argument by replacing the string XXX with the
number of the file, starting at 0.
"""

BLACKLIST = 'config/mal_blacklist.txt'
COLSTATS = 'config/tpch_sf100_stats.txt'

def main(args):
    blacklist = Utils.init_blacklist(BLACKLIST)
    col_stats = ColumnStatsD.fromFile(COLSTATS)

    sys.stdout.flush()
    dataset = MalDictionary.fromJsonFile(
        args.input_file,
        blacklist,
        col_stats
    )
    tags = sorted(dataset.query_tags)
    if args.limit:
        tags = tags[:args.limit]

    tag_map_file = open(args.tag_map, 'a') if args.tag_map else None

    counter = 0
    for tag in tags:
        out_name = args.output_files.replace('XXX', '%03d' % counter)
        short_name = os.path.basename(out_name)
        if '.' in short_name:
            short_name = short_name[:short_name.index('.')]

        if tag_map_file:
            tag_map_file.write('{}:{}\n'.format(tag, short_name))
        counter += 1
        contents = dataset.filter(lambda x: x.tag == tag)
        with open(out_name, 'wb') as f:
            if out_name.endswith('.lz4'):
                f = lz4frame.LZ4FrameFile(f, mode='wb')
            sys.stderr.write('\r[{}/{}] tag {}'.format(counter, len(tags), tag))
            pickle.dump(contents, f)
            f.close()
    sys.stderr.write('\rDone                       \n')
    if tag_map_file:
        tag_map_file.close()

def arg_parser():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('input_file',
        help='File containing the traces')
    parser.add_argument('output_files',
        help='Pattern for the names of the output files.  XXX is replaced with the file number')
    parser.add_argument('--limit', type=int,
        help='Write only the first N pickles')
    parser.add_argument('--tag-map',
        help='Write tag <-> name mapping to this file')
    return parser


if __name__ == "__main__":
    args = arg_parser().parse_args()
    # print(args)
    sys.exit(main(args))
