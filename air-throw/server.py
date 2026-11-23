"""The room: a small web server on your Mac that every device joins over Wi-Fi.

- Phones and laptops open http://<your-mac>:8765 in a browser (no app needed).
- Everyone connected over a WebSocket sees the same list of people.
- A throw is sent to one person; pictures travel as files the server keeps in items/.
The Mac app (main.py) talks to the room through a few thread-safe methods.
"""
import asyncio
import json
import os
import queue
import socket
import threading
import time
import uuid

from aiohttp import WSMsgType, web

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")
ITEMS = os.path.join(HERE, "items")
PORT = 8765
COLORS = ["#c98b8b", "#8fa8c9", "#a9b98c", "#c9a77d", "#a58cc0", "#7fb5ad", "#c48fae", "#b0a089"]


def local_ip():
    """This Mac's address on the Wi-Fi (what other devices type to reach it)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))   # no packet is actually sent
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class Room:
    def __init__(self, host_name):
        os.makedirs(ITEMS, exist_ok=True)
        self.host_name = host_name
        self.clients = {}              # id -> {"ws": ws, "name": str, "color": str}
        self.inbox = queue.Queue()     # things thrown to the Mac: (from_id, from_name, item)
        self.roster_version = 0
        self.loop = None
        self.url = f"http://{local_ip()}:{PORT}"

    # ---------------------------------------------------------- for the Mac app (any thread)
    def people(self):
        """Everyone you can throw to (not including the Mac itself)."""
        return [{"id": cid, "name": c["name"], "color": c["color"]} for cid, c in list(self.clients.items())]

    def throw_from_host(self, to_id, item, vx=0.0, vy=-1.0, spin=0.0):
        if self.loop is None:
            return
        msg = {"type": "incoming", "from": "host", "from_name": self.host_name, "item": item,
               "vx": vx, "vy": vy, "spin": spin}
        asyncio.run_coroutine_threadsafe(self._send(to_id, msg), self.loop)

    def save_item_image(self, png_bytes, ext="png"):
        name = f"{uuid.uuid4().hex[:12]}.{ext}"
        with open(os.path.join(ITEMS, name), "wb") as f:
            f.write(png_bytes)
        return "/items/" + name

    # ---------------------------------------------------------- server internals
    async def _send(self, cid, msg):
        c = self.clients.get(cid)
        if c is not None:
            try:
                await c["ws"].send_json(msg)
            except Exception:
                pass

    async def _broadcast_roster(self):
        self.roster_version += 1
        everyone = [{"id": "host", "name": self.host_name, "color": "#d9b77e"}] + self.people()
        for cid in list(self.clients):
            others = [p for p in everyone if p["id"] != cid]
            await self._send(cid, {"type": "roster", "people": others, "you": cid})

    async def ws_handler(self, request):
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=8 * 1024 * 1024)
        await ws.prepare(request)
        cid = uuid.uuid4().hex[:8]
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except ValueError:
                    continue
                kind = data.get("type")
                if kind == "join":
                    name = (data.get("name") or "friend").strip()[:24] or "friend"
                    self.clients[cid] = {"ws": ws, "name": name, "color": COLORS[len(self.clients) % len(COLORS)]}
                    await self._broadcast_roster()
                elif kind == "throw" and cid in self.clients:
                    to = data.get("to")
                    item = data.get("item") or {}
                    sender = self.clients[cid]["name"]
                    if to == "host":
                        self.inbox.put((cid, sender, item))
                    else:
                        await self._send(to, {"type": "incoming", "from": cid, "from_name": sender, "item": item,
                                              "vx": data.get("vx", 0), "vy": data.get("vy", -1),
                                              "spin": data.get("spin", 0)})
        finally:
            if self.clients.pop(cid, None) is not None:
                await self._broadcast_roster()
        return ws

    async def upload(self, request):
        """A phone uploads a photo (or its profile picture) before throwing it."""
        reader = await request.multipart()
        part = await reader.next()
        if part is None:
            return web.json_response({"error": "no file"}, status=400)
        ext = os.path.splitext(part.filename or "")[1].lower().lstrip(".") or "jpg"
        if ext not in ("jpg", "jpeg", "png", "gif", "webp", "heic"):
            ext = "jpg"
        data = await part.read(decode=False)
        if len(data) > 25 * 1024 * 1024:
            return web.json_response({"error": "too big"}, status=413)
        return web.json_response({"url": self.save_item_image(data, ext)})

    async def index(self, request):
        return web.FileResponse(os.path.join(WEB, "index.html"), headers={"Cache-Control": "no-store"})

    def app(self):
        app = web.Application(client_max_size=30 * 1024 * 1024)
        app.router.add_get("/", self.index)
        app.router.add_get("/ws", self.ws_handler)
        app.router.add_post("/upload", self.upload)
        app.router.add_static("/items/", ITEMS)
        app.router.add_static("/web/", WEB)
        return app

    def start(self):
        """Run the server on a background thread."""
        ready = threading.Event()

        def run():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            runner = web.AppRunner(self.app())
            self.loop.run_until_complete(runner.setup())
            site = web.TCPSite(runner, "0.0.0.0", PORT)
            self.loop.run_until_complete(site.start())
            ready.set()
            self.loop.run_forever()

        threading.Thread(target=run, daemon=True).start()
        ready.wait(5)
        time.sleep(0.05)
        return self.url
