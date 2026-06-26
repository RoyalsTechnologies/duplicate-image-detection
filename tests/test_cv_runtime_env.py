import os


from app.computer_vision import runtime_env


def test_configure_cv_runtime_env_sets_hf_home(monkeypatch) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("YOLO_CONFIG_DIR", raising=False)

    runtime_env.configure_cv_runtime_env()

    assert runtime_env.settings.cv_cache_dir.as_posix() in os.environ["HF_HOME"]
    assert os.environ["HUGGINGFACE_HUB_CACHE"].endswith("/huggingface/hub")
