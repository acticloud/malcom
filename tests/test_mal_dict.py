import os
import pytest
import yaml

from malcom import mal_dict
from malcom import utils
from malcom import stats


@pytest.fixture(params=[
    'config/tpch_sf100.yaml'
])
def blist_stats(request):
    parent_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
    cfilename = os.path.join(
        parent_directory,
        request.param
    )
    with open(cfilename) as conf_file:
        configuration = yaml.safe_load(conf_file)
    print(configuration)
    root_path = configuration['root_path']
    blist = utils.Utils.init_blacklist(
        os.path.join(
            root_path,
            configuration['blacklist']
        )
    )

    cstats = stats.ColumnStatsD.fromFile(
        os.path.join(
            root_path,
            configuration['stats']
        )
    )

    return (blist, cstats)


class TestMalDict():
    def test_from_json_uncompressed(self, blist_stats):
        fn = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data/test_uncompressed_trace.json'
        )

        d = mal_dict.MalDictionary.fromJsonFile(fn, blist_stats[0], blist_stats[1])
        assert len(d.query_tags) > 0

    def test_from_json_compressed(self, blist_stats):
        fn = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data/test_compressed_trace.json.gz'
        )

        d = mal_dict.MalDictionary.fromJsonFile(fn, blist_stats[0], blist_stats[1])
        assert len(d.query_tags) > 0
