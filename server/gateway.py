from __future__ import annotations

import asyncio
import os
import secrets
import signal
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
LEASE_SECONDS = float(os.getenv('SAMOTATOR_LEASE_SECONDS', '60'))
def max_sessions_from_env() -> int:
    value = os.getenv('SAMOTATOR_MAX_SESSIONS', '8')
    try:
        limit = int(value)
    except ValueError as error:
        raise ValueError('SAMOTATOR_MAX_SESSIONS must be a positive integer.') from error
    if limit < 1:
        raise ValueError('SAMOTATOR_MAX_SESSIONS must be a positive integer.')
    return limit


MAX_SESSIONS = max_sessions_from_env()
WORKER_PORT = int(os.getenv('SAMOTATOR_WORKER_PORT', '18001'))


class Sessions:
    def __init__(self, timeout: float = LEASE_SECONDS, limit: int = MAX_SESSIONS):
        if limit < 1:
            raise ValueError('Session limit must be a positive integer.')
        self.timeout = timeout
        self.limit = limit
        self.users: dict[str, float] = {}
        self.reservations: dict[str, str] = {}

    def expire(self):
        now = time.monotonic()
        for token, seen in list(self.users.items()):
            if now - seen > self.timeout:
                self.remove(token)

    def remove(self, token: str):
        self.users.pop(token, None)
        self.reservations = {image: owner for image, owner in self.reservations.items() if owner != token}

    def connect(self, token: str | None):
        self.expire()
        if token not in self.users:
            if len(self.users) >= self.limit:
                raise HTTPException(503, f'All {self.limit} annotation places are occupied.')
            token = secrets.token_urlsafe(32)
        assert token is not None
        self.users[token] = time.monotonic()
        return token

    def require(self, token: str | None):
        self.expire()
        if not token or token not in self.users:
            raise HTTPException(401, 'Session expired. Reconnect to continue.')
        self.users[token] = time.monotonic()
        return token

    def reserve(self, token: str, image: str | None):
        self.reservations = {key: owner for key, owner in self.reservations.items() if owner != token or key == image}
        if image is None:
            return False
        owner = self.reservations.get(image)
        if owner is not None and owner != token:
            return False
        self.reservations[image] = token
        return True


sessions = Sessions()
lock = asyncio.Lock()
worker: asyncio.subprocess.Process | None = None
worker_sessions: set[str] = set()
client: httpx.AsyncClient

async def start_worker():
    global worker
    if worker is not None and worker.returncode is None:
        return
    worker = await asyncio.create_subprocess_exec(sys.executable, '-m', 'uvicorn', 'server.app:app', '--host', '127.0.0.1', '--port', str(WORKER_PORT), '--workers', '1')

async def stop_worker():
    global worker
    if worker is None:
        return
    process, worker = worker, None
    if process.returncode is None:
        process.send_signal(signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=30)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

async def remove_worker_session(token: str):
    if worker is not None and worker.returncode is None:
        try:
            await client.post(f'http://127.0.0.1:{WORKER_PORT}/internal/session/remove', headers={'x-session-token': token}, timeout=2)
        except httpx.HTTPError:
            pass

async def reap():
    while True:
        await asyncio.sleep(min(5, max(1, LEASE_SECONDS / 4)))
        async with lock:
            sessions.expire()
            for token in worker_sessions - set(sessions.users):
                await remove_worker_session(token)
                worker_sessions.discard(token)
            if not sessions.users:
                await stop_worker()

@asynccontextmanager
async def lifespan(_: FastAPI):
    global client
    client = httpx.AsyncClient(timeout=httpx.Timeout(120, connect=2))
    task = asyncio.create_task(reap())
    yield
    task.cancel()
    async with lock:
        await stop_worker()
    await client.aclose()

app = FastAPI(title='Samotator gateway', lifespan=lifespan)

class Connect(BaseModel):
    token: str | None = None

class Reservation(BaseModel):
    imageId: str | None = None

@app.post('/api/session/connect')
async def connect(request: Connect):
    async with lock:
        previous = set(sessions.users)
        token = sessions.connect(request.token)
        for expired in previous - set(sessions.users):
            await remove_worker_session(expired)
            worker_sessions.discard(expired)
        worker_sessions.add(token)
        await start_worker()
        return {'token': token, 'connected': len(sessions.users), 'limit': sessions.limit, 'leaseSeconds': LEASE_SECONDS}

@app.post('/api/session/heartbeat')
async def heartbeat(request: Request):
    async with lock:
        token = sessions.require(request.headers.get('x-session-token'))
        if worker is None or worker.returncode is not None:
            await start_worker()
        return {'connected': len(sessions.users), 'limit': sessions.limit, 'reservation': next((image for image, owner in sessions.reservations.items() if owner == token), None)}

@app.post('/api/session/disconnect')
async def disconnect(request: Request):
    async with lock:
        token = sessions.require(request.headers.get('x-session-token'))
        sessions.remove(token)
        await remove_worker_session(token)
        worker_sessions.discard(token)
        if not sessions.users:
            await stop_worker()
        return {'connected': len(sessions.users)}

@app.post('/api/session/reservation')
async def reserve(request: Request, body: Reservation):
    async with lock:
        token = sessions.require(request.headers.get('x-session-token'))
        return {'imageId': body.imageId, 'editable': sessions.reserve(token, body.imageId)}

@app.api_route('/api/{path:path}', methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE'])
async def proxy(path: str, request: Request):
    # Serialize saves against reservation release and expiry; ordinary inference does not hold the lock.
    async with lock:
        token = sessions.require(request.headers.get('x-session-token'))
        parts = path.split('/')
        saving = request.method == 'PUT' and len(parts) == 3 and parts[0] == 'images' and parts[2] == 'annotations'
        if saving and sessions.reservations.get(parts[1]) != token:
            raise HTTPException(409, 'This image is not reserved by your session.')
        if saving:
            return await forward(path, request, token)
    return await forward(path, request, token)

async def forward(path: str, request: Request, token: str):
    body = await request.body()
    headers = {'x-session-token': token}
    if request.headers.get('content-type'):
        headers['content-type'] = request.headers['content-type']
    response = None
    for _ in range(300):
        try:
            response = await client.request(request.method, f'http://127.0.0.1:{WORKER_PORT}/api/{path}', params=request.query_params, content=body, headers=headers)
            break
        except httpx.ConnectError:
            await asyncio.sleep(0.1)
    if response is None:
        raise HTTPException(503, 'GPU worker is starting or unavailable.')
    excluded = {'content-length', 'transfer-encoding', 'connection', 'content-encoding'}
    return Response(response.content, status_code=response.status_code, headers={k: v for k, v in response.headers.items() if k.lower() not in excluded})

@app.get('/data/{path:path}')
async def image_file(path: str):
    from .manifest import IMAGE_EXTENSIONS
    relative = Path(path)
    root = (ROOT / 'data').resolve()
    candidate = (root / relative).resolve()
    if (any(part.startswith('.') for part in relative.parts) or not candidate.is_relative_to(root)
            or candidate.suffix.lower() not in IMAGE_EXTENSIONS or not candidate.is_file()):
        raise HTTPException(404, 'Image not found.')
    return FileResponse(candidate)
if (ROOT / 'dist').is_dir():
    app.mount('/assets', StaticFiles(directory=ROOT / 'dist' / 'assets'), name='assets')

@app.get('/{path:path}')
async def frontend(path: str):
    candidate = (ROOT / 'dist' / path).resolve()
    if candidate.is_relative_to(ROOT / 'dist') and candidate.is_file():
        return FileResponse(candidate)
    index = ROOT / 'dist' / 'index.html'
    if not index.is_file():
        raise HTTPException(503, 'Frontend not built. Run npm run build.')
    return FileResponse(index)
