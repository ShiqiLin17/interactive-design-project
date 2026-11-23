# 🎄 Christmas Fog Window

A realistic winter window: condensation with real-looking water droplets, frost creeping in from the edges, snow piled on the sill outside, warm firelight on one side and cool blue winter light on the other. Blow on the glass to fog it up, wipe it with your hand, write in the fog, and decorate it with Christmas stickers using hand gestures.

![preview](preview.png)
*(Preview made with a placeholder instead of a real camera feed. With your camera, you'll be the one behind the glass.)*

## How to play

| Do this | What happens |
|---|---|
| 😮 **Open your mouth and blow** at the camera | The window fogs up, spreading out from your mouth |
| ✋ **Open palm** (either hand) | Wipes the fog off wherever your hand moves |
| 🤏 **Sticker hand:** thumb + **index** (each fingertip shows a mini preview of its sticker) | 🎄 Christmas tree pops out |
| thumb + **middle** | ⛄ Snowman |
| thumb + **ring** | 🎁 Gift |
| thumb + **pinky** | 🍬 Candy cane |
| 🤏 **Other hand:** thumb + **index** on a sticker | Grab it and move it. Bring your hand **closer** to the camera to make it bigger, or **farther** to make it smaller. Let go to place it. |
| ✍️ **Other hand:** thumb + **middle** | Draw in the fog with your fingertip and write your own words |
| Drag a sticker to the circle in the corner | Removes it |

The sticker hand is your **right** hand by default. Press **S** to swap. Both hands show small previews on their fingertips: mini stickers on the sticker hand, and a move icon (index) and pencil icon (middle) on the other hand.

Tip: after you draw, the open-palm wipe pauses for a moment so you can lift your hand between letters without erasing them.

**Keys:** `S` swap hands · `R` fog the whole window · `W` clear all fog · `Z` undo · `C` clear stickers · `P` save a photo · `H` hide/show help · `F` fullscreen · `Q` quit

## Run it

```bash
cd christmas-fog-window
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

The first run downloads two small MediaPipe models (hand + face tracking) into this folder.

## Troubleshooting

- **Black screen or "Couldn't open the camera" on Mac:** go to System Settings → Privacy & Security → Camera and turn on access for Terminal (or VS Code). Then restart Terminal.
- **Wrong camera:** change `CAMERA_INDEX = 0` to `1` at the top of `main.py`.
- **`pip` can't find mediapipe:** your Python version is probably too new. Use Python 3.10–3.12, for example `python3.12 -m venv .venv`.
- **Fog triggers when you're just talking:** raise `MOUTH_OPEN` at the top of `main.py` (try `0.07`).
- **Want the fog to slowly come back after wiping:** set `FOG_REGROW_SECONDS = 25`.

## How it works

- **MediaPipe Face Landmarker** tracks your lips. When the gap between them is big enough, fog spreads out from your mouth.
- **MediaPipe Hand Landmarker** tracks 21 points on each hand. Fingertip distances detect pinches, and an open palm with the thumb spread acts as the wiper. Hand size on screen (how close you are) controls sticker zoom.
- **Fog** is a mask over the view: heavily blurred, lightened, and tinted cool blue-white, glowing warm near the fire. Wiping and drawing cut clear lines into the mask.
- **Droplets** each work like a tiny lens. They show a flipped, sharper view of what's behind the glass, with a bright highlight, a dark rim, and light focused at the bottom. Some drops slide down and leave clear trails in the fog.
- **The window** is drawn in its own space, then warped in perspective so the left side is closer and the right side is cropped. It has frost grains on the pane edges, granular snow on the outside sill, falling snow and soft fairy lights outside, and firelight that flickers on the frame and reflects in the glass.
- Everything is drawn with code (OpenCV + NumPy). There are no image files.

## Files

- `main.py`: camera loop, fog, gestures, stickers
- `window.py`: the window frame, frost, snow, lighting and the world outside
- `droplets.py`: water droplets and sliding drops
- `stickers.py`: the Christmas stickers
- `text.py`: nicer fonts for the on-screen text
