from server import run


def test_nvidia_library_path_uses_active_interpreter_paths(tmp_path, monkeypatch):
    venv = tmp_path / 'venv' / 'lib' / 'python' / 'site-packages'
    user = tmp_path / 'user-site'
    for site in (venv, user):
        (site / 'nvidia' / 'cudnn' / 'lib').mkdir(parents=True)
    (venv / 'nvidia' / 'cublas' / 'lib').mkdir(parents=True)
    monkeypatch.setattr(run.sys, 'path', [str(venv), str(user), str(venv)])
    monkeypatch.setenv('LD_LIBRARY_PATH', '/custom/lib')
    assert run.nvidia_library_path().split(':') == [
        str(venv / 'nvidia' / 'cublas' / 'lib'),
        str(venv / 'nvidia' / 'cudnn' / 'lib'),
        str(user / 'nvidia' / 'cudnn' / 'lib'),
        '/custom/lib',
    ]
