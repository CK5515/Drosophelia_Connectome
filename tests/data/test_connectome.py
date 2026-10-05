import numpy as np
import pandas as pd
import pytest

from flybrain.data.connectome import build_connectome, load_connectome, save_connectome


@pytest.fixture
def tiny_files(tmp_path):
    ids = [720575940000000010, 720575940000000020, 720575940000000030]
    pd.DataFrame({"Completed": [True, True, True]}, index=ids).to_csv(tmp_path / "comp.csv")
    pd.DataFrame({
        "Presynaptic_ID": [ids[0], ids[1], ids[2]],
        "Postsynaptic_ID": [ids[1], ids[2], ids[0]],
        "Presynaptic_Index": [0, 1, 2],
        "Postsynaptic_Index": [1, 2, 0],
        "Connectivity": [2, 3, 1],
        "Excitatory": [1, -1, 1],
        "Excitatory x Connectivity": [2, -3, 1],
    }).to_parquet(tmp_path / "conn.parquet")
    return tmp_path / "conn.parquet", tmp_path / "comp.csv", ids


def test_build_connectome_signed_directed_weights(tiny_files):
    conn_path, comp_path, ids = tiny_files
    c = build_connectome(conn_path, comp_path)
    expected = np.array([[0, 2, 0], [0, 0, -3], [1, 0, 0]], dtype=np.float32)
    np.testing.assert_array_equal(c.W.toarray(), expected)
    assert c.W.dtype == np.float32
    assert c.n == 3
    np.testing.assert_array_equal(c.flywire_ids, ids)


def test_index_of_maps_ids_and_rejects_unknown(tiny_files):
    c = build_connectome(*tiny_files[:2])
    ids = tiny_files[2]
    np.testing.assert_array_equal(c.index_of([ids[2], ids[0]]), [2, 0])
    with pytest.raises(KeyError, match="not in connectome"):
        c.index_of([123])


def test_save_load_roundtrip(tiny_files, tmp_path):
    c = build_connectome(*tiny_files[:2])
    save_connectome(c, tmp_path / "cache.npz")
    d = load_connectome(tmp_path / "cache.npz")
    np.testing.assert_array_equal(c.W.toarray(), d.W.toarray())
    np.testing.assert_array_equal(c.flywire_ids, d.flywire_ids)
