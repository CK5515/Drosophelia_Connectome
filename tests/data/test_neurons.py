import pytest

from flybrain.data.neurons import INPUT_GROUPS, MN9_IDS
from flybrain.paths import RAW


def test_group_order_and_sizes():
    assert list(INPUT_GROUPS) == ["sugar_R", "sugar_L", "bitter", "water", "ir94e"]
    assert [len(v) for v in INPUT_GROUPS.values()] == [21, 10, 21, 18, 18]


def test_no_duplicates_anywhere():
    everything = [i for ids in INPUT_GROUPS.values() for i in ids] + list(MN9_IDS)
    assert len(everything) == len(set(everything)) == 90


@pytest.mark.realdata
def test_all_ids_exist_in_v630():
    if not (RAW / "2023_03_23_completeness_630_final.csv").exists():
        pytest.skip("raw data not downloaded")
    from flybrain.data.connectome import load_or_build_connectome
    from flybrain.data.fetch import fetch_raw
    paths = fetch_raw()
    conn = load_or_build_connectome(paths["connectivity"], paths["completeness"])
    for ids in INPUT_GROUPS.values():
        conn.index_of(ids)
    conn.index_of(MN9_IDS)
