# Air Throw

Grab things in the air and throw them to your friends' phones and laptops. Your friends can catch them and throw them on to someone else.

You can throw **stickers**, **photos** and your **profile card** (Instagram, Xiaohongshu, LinkedIn, phone).

## How it works

- **Your Mac** runs the app: your camera, with your beauty filter, plus the things you can throw. It also hosts the "room".
- **Friends join** by scanning the QR code in the corner of your screen with their phone camera. It opens a web page, so there's no app to install. It works on iPhone, Android, laptops and tablets. Everyone just needs to be on the **same Wi‑Fi**.
- Everyone in the room shows up as a name along the top. **Throw toward a name** to send something to that person.

## On the Mac

| Do this | What happens |
|---|---|
| 🤏 Pinch **thumb + index** on something | Grab it and move it around |
| Flick your hand **up toward a name** while pinching, then let go | Throw it to that person |
| **S** | Add the next sticker (your watercolor stickers) |
| **P** | Snap a photo with the camera (filter on) |
| **C** | Add your profile card (edit `profile.json`; put a photo next to it as `me.jpg`) |
| Drop pictures in `my_photos/` | They appear on screen, ready to throw |
| **X** | Remove the last thing |
| **V** | Switch camera (remembered) |
| **F** | Fullscreen |
| **Q** | Quit |

Things thrown to you fly in from that person's name and land on your screen. Photos people throw you are saved in `received/`.

## On a phone or laptop

- **Swipe something up toward a name** to throw it.
- **+ photo** adds a picture from your camera roll.
- **my card** adds your profile card. The first time, you fill it in. Right-click or long-press the button to edit it later.
- **Tap** something to open it: full-size photos (press and hold to save to Photos), or card links like *open Instagram*.
- **catch mode:** turn it on to make things fly across your screen. Tap to catch them, or they go to the *missed* pile.

## Run it

```bash
cd air-throw
conda activate xmas
pip install -r requirements.txt
python main.py
```

The first time, macOS may ask whether Python can accept incoming network connections. Click **Allow**, so friends' phones can connect.

## Files

- `main.py`: the Mac app (camera, hand tracking, grabbing and throwing)
- `server.py`: the room (web server + live connection to every device)
- `web/index.html`: the page friends open on their phones and laptops
- Uses Shiqi's beauty filter, fonts and stickers from `../christmas-fog-window/`
