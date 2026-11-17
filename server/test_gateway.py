import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from server import gateway


def test_session_limit_reconnect_expiry_and_reservations():
    sessions = gateway.Sessions(timeout=10)
    tokens = [sessions.connect(None) for _ in range(8)]
    assert len(set(tokens)) == 8
    assert sessions.connect(tokens[0]) == tokens[0]
    with pytest.raises(HTTPException) as full:
        sessions.connect(None)
    assert full.value.status_code == 503
    assert sessions.reserve(tokens[0], 'image')
    assert not sessions.reserve(tokens[1], 'image')
    sessions.users[tokens[0]] = time.monotonic() - 11
    sessions.expire()
    assert sessions.reserve(tokens[1], 'image')
    assert sessions.connect(None) not in tokens
    sessions.remove(tokens[1])
    assert 'image' not in sessions.reservations


def test_api_reservations_and_worker_lifetime(monkeypatch):
    monkeypatch.setattr(gateway, 'sessions', gateway.Sessions())
    gateway.worker_sessions.clear()
    lifecycle = []
    async def start(): lifecycle.append('start')
    async def stop(): lifecycle.append('stop')
    async def remove(token): lifecycle.append(('remove', token))
    monkeypatch.setattr(gateway, 'start_worker', start)
    monkeypatch.setattr(gateway, 'stop_worker', stop)
    monkeypatch.setattr(gateway, 'remove_worker_session', remove)
    with TestClient(gateway.app) as client:
        a = client.post('/api/session/connect', json={}).json()['token']
        b = client.post('/api/session/connect', json={}).json()['token']
        ah = {'X-Session-Token': a}
        bh = {'X-Session-Token': b}
        assert client.post('/api/session/reservation', headers=ah, json={'imageId': 'im'}).json()['editable']
        assert not client.post('/api/session/reservation', headers=bh, json={'imageId': 'im'}).json()['editable']
        assert client.put('/api/images/im/annotations', headers=bh, json={}).status_code == 409
        assert client.post('/api/session/disconnect', headers=ah).status_code == 200
        assert 'stop' not in lifecycle
        assert client.post('/api/session/reservation', headers=bh, json={'imageId': 'im'}).json()['editable']
        assert client.post('/api/session/disconnect', headers=bh).status_code == 200
        assert lifecycle.count('stop') == 1


@pytest.mark.parametrize('value,expected', [('1', 1), ('8', 8), ('12', 12)])
def test_session_limit_setting(monkeypatch, value, expected):
    monkeypatch.setenv('SAMOTATOR_MAX_SESSIONS', value)
    assert gateway.max_sessions_from_env() == expected


@pytest.mark.parametrize('value', ['', '0', '-1', '1.5', 'many'])
def test_session_limit_setting_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv('SAMOTATOR_MAX_SESSIONS', value)
    with pytest.raises(ValueError, match='SAMOTATOR_MAX_SESSIONS must be a positive integer'):
        gateway.max_sessions_from_env()


def test_default_session_limit_setting(monkeypatch):
    monkeypatch.delenv('SAMOTATOR_MAX_SESSIONS', raising=False)
    assert gateway.max_sessions_from_env() == 8


def test_configured_limit_is_returned_by_session_api(monkeypatch):
    monkeypatch.setattr(gateway, 'sessions', gateway.Sessions(limit=2))
    gateway.worker_sessions.clear()
    async def noop(*args): pass
    monkeypatch.setattr(gateway, 'start_worker', noop)
    monkeypatch.setattr(gateway, 'stop_worker', noop)
    monkeypatch.setattr(gateway, 'remove_worker_session', noop)
    with TestClient(gateway.app) as client:
        first = client.post('/api/session/connect', json={}).json()
        assert first['limit'] == 2
        assert client.post('/api/session/connect', json={}).json()['limit'] == 2
        response = client.post('/api/session/connect', json={})
        assert response.status_code == 503
        assert '2 annotation places' in response.json()['detail']
        headers = {'X-Session-Token': first['token']}
        assert client.post('/api/session/heartbeat', headers=headers).json()['limit'] == 2
        assert client.post('/api/session/connect', json={'token': first['token']}).json()['token'] == first['token']
        assert client.post('/api/session/disconnect', headers=headers).status_code == 200
        assert client.post('/api/session/connect', json={}).status_code == 200
