from crowd_monitoring.config import load_config, pipeline_kwargs_from_config, risk_config_from_dict


def test_default_config_loads():
    cfg = load_config()
    assert "model" in cfg
    assert "risk" in cfg
    assert "clustering" in cfg


def test_pipeline_kwargs_from_default_config():
    cfg = load_config()
    kwargs = pipeline_kwargs_from_config(cfg)
    assert kwargs["model_name"] == "DM-Count"
    assert kwargs["model_weights"] == "QNRF"
    assert kwargs["eps_px"] == 40
    assert kwargs["risk_config"].count_warning == 80


def test_risk_config_from_partial_dict():
    rc = risk_config_from_dict({"risk": {"count_caution": 5}})
    assert rc.count_caution == 5
    assert rc.count_warning == 80  # untouched default
