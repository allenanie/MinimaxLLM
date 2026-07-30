import pytest

import store


def test_artifact_id_stable_under_key_order(tmp_path):
    a = store.put_artifact(tmp_path, {"a.py": "print(1)\n", "b.py": "print(2)\n"})
    b = store.put_artifact(tmp_path, {"b.py": "print(2)\n", "a.py": "print(1)\n"})
    assert a == b
    assert [p.name for p in (tmp_path / "artifacts").iterdir()] == [a]


def test_append_step_refuses_overwrite(tmp_path):
    assert store.append_step(tmp_path, {"selected": "a"}) == 0
    # A gap makes the count point at an occupied slot (miscounted resume / concurrent writer).
    occupied = tmp_path / "steps" / "0002.json"
    occupied.write_text('{"selected": "a"}')
    with pytest.raises(FileExistsError):
        store.append_step(tmp_path, {"selected": "b"})
    assert occupied.read_text() == '{"selected": "a"}'
