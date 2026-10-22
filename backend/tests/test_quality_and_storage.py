import numpy as np
import pytest

from touchline import quality
from touchline.agent.tools import restrict
from touchline.storage import LocalStorage, S3Storage


def seg(frames, tracks, stride=6, mc=False):
    return {"segment_id": "seg1", "start": 0, "end": 120, "config": {"stride": stride, "motion_compensation": mc},
            "frames": frames, "tracks": tracks, "tracker_stats": {"assoc_residual_median_h": 0.1}}


def frame(i, kept=10, grass=10, out=0, drift=0.0, align=0.9):
    return {"frame": i, "t": i / 30, "n_kept": kept, "n_on_grass": grass, "n_out_of_pitch": out, "drift_px": drift,
            "line_alignment": align, "motion": {"inlier_ratio": 0.9, "reproj_error_px": 1.0, "validated": True}}


def track(frames, xs):
    return {"tid": "t", "obs": [{"frame": f, "t": f / 30, "pitch": [x, 30.0]} for f, x in zip(frames, xs)]}


def test_signals_measured_values():
    frames = [frame(i, out=1 if i == 0 else 0, drift=5.0 * i / 6) for i in range(0, 120, 6)]
    tracks = [track(range(0, 120, 6), np.linspace(10, 20, 20)),  # 10 m in 3.8 s: plausible
              track([0, 6], [10, 40])]  # 30 m in 0.2 s: implausible, and short
    q = quality.assess(seg(frames, tracks), np.array([[0, 13.84], [16.5, 13.84], [16.5, 54.16], [0, 54.16]]), True)
    s = q.signals
    assert s["out_of_pitch_fraction"] == pytest.approx(1 / 200)
    assert s["implausible_jump_rate"] == pytest.approx(1 / 20)
    assert s["short_track_ratio"] == pytest.approx(0.5)
    assert s["fragmentation_index"] == pytest.approx(2 / 10)
    assert s["camera_drift_px"] == pytest.approx(5.0 * 114 / 6)
    assert "camera_drift_px" in q.breaches


def test_metric_signals_unavailable_without_calibration():
    q = quality.assess(seg([frame(0), frame(6)], [track([0, 6], [1, 2])]), None, False)
    assert q.signals["out_of_pitch_fraction"] is None
    assert q.breaches[0] == "calibration_valid"


def test_drift_not_a_breach_once_compensated():
    q = quality.assess(seg([frame(0, drift=80), frame(6, drift=90)], [], mc=True), None, True)
    assert "camera_drift_px" not in q.breaches


def test_restrict_uses_identical_frames():
    s = seg([frame(i) for i in range(0, 12, 2)], [track(range(0, 12, 2), range(6))], stride=2)
    r = restrict(s, {0, 6})
    assert [f["frame"] for f in r["frames"]] == [0, 6]
    assert [o["frame"] for o in r["tracks"][0]["obs"]] == [0, 6]


def test_every_signal_has_definition_units_threshold():
    for spec in quality.spec_table():
        assert spec["definition"] and spec["unit"] and spec["bad_when"] in ("above", "below")


def test_local_storage_roundtrip_and_traversal(tmp_path):
    st = LocalStorage(tmp_path)
    st.put_bytes("jobs/a/report.json", b"{}")
    assert st.exists("jobs/a/report.json") and st.read_bytes("jobs/a/report.json") == b"{}"
    with pytest.raises(ValueError):
        st.put_bytes("../escape.txt", b"x")


def test_s3_storage_encrypts_and_uses_prefix(tmp_path):
    boto3 = pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")
    with moto.mock_aws():
        c = boto3.client("s3", region_name="us-east-1")
        c.create_bucket(Bucket="tl-test")
        st = S3Storage("tl-test", "touchline/", "us-east-1", tmp_path, client=c)
        st.put_bytes("jobs/x/trace.jsonl", b"{}\n", "application/x-ndjson")
        head = c.head_object(Bucket="tl-test", Key="touchline/jobs/x/trace.jsonl")
        assert head["ServerSideEncryption"] == "AES256"
        (tmp_path / "s3cache" / "jobs/x/trace.jsonl").unlink()
        assert st.read_bytes("jobs/x/trace.jsonl") == b"{}\n"
        with pytest.raises(ValueError):
            st.put_bytes("../../etc", b"x")
