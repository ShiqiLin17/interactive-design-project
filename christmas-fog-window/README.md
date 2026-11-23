# 🎄 Christmas Fog Window

A realistic winter window: condensation with real-looking water droplets, frost creeping in from the edges, snow piled on the sill outside, warm firelight on one side and cool blue winter light on the other. Blow on the glass to fog it up, wipe it with your hand, write in the fog, and decorate it with elegant Christmas ornaments using hand gestures.

![preview](preview.png)
*(Preview made with a placeholder instead of a real camera feed. With your camera, you'll be the one behind the glass.)*

## How to play

When it starts, the glass is clear and it says *breathe on it...*

| Do this | What happens |
|---|---|
| 😮 **Make an "O" with your lips and blow** at the camera (smiling or talking won't fog it) | The window fogs up, spreading out from your mouth |
| ✊ **Closed fist** (sticker hand only) | Wipe the fog with the pinky side of your fist, like wiping a real window |
| 🤏 **Ornament hand:** thumb + **index** (each fingertip shows a mini preview) | A glossy red glass ornament appears |
| thumb + **middle** | A silver snowflake |
| thumb + **ring** | A faceted gold star |
| thumb + **pinky** | A holly sprig with berries |
| 🤏 **Other hand:** thumb + **middle** on an ornament | Grab it and move it. Bring your hand **closer** to the camera to make it bigger, or **farther** to make it smaller. Let go to place it. |
| ✨ **Other hand:** thumb + **ring** | Switch your face accessory |
| ✍️ **Other hand:** turn it **sideways** (pinky side toward the screen, like holding a pen) and pinch **thumb + index** | Draw in the fog with your fingertip and write your own words |
| Drag an ornament to the circle in the corner | Removes it |

The ornament hand is your **right** hand by default. Press **S** to swap. Both hands show small previews on their fingertips: mini ornaments on that hand, and a pencil icon (index) and move icon (middle) on the other hand.

Tip: after you draw, fist-wiping pauses for a moment so moving your hand between letters doesn't erase them.

**Keys:** `V` switch camera (laptop / iPhone) · `N` next page of stickers · `E` switch face filter · `S` swap hands · `R` fog the whole window · `W` clear all fog · `Z` undo · `C` clear stickers · `P` save a photo · `H` show help again (it fades after a few seconds) · `F` fullscreen · `Q` quit

## Your own stickers

Put transparent PNGs in a folder called `my_stickers/` inside this folder, and they replace the built-in ornaments automatically, in filename order (`01.png`, `02.png`, ...). Your sticker hand holds 4 at a time, one per finger, and the fingertip previews show which is which. Press **N** to flip to the next 4.

Have a sticker *sheet* on a white background instead? Cut it into separate stickers with:

```bash
python cut_stickers.py path/to/sheet.jpg
```

The stickers in `my_stickers/` are Shiqi's own artwork.

## Face filters

Your face gets a Douyin-style beauty filter all the time, plus a Christmas accessory you can switch.

**Beauty (always on), tuned to match Shiqi's favorite Douyin filter:**
- Face shaping: an exact copy of how her Douyin filter reshapes her face (smaller face, slimmer jaw and chin, narrower nose, eyes that look bigger, shorter face and lower hairline). It was measured from her before/after video by tracking all 468 face points at the moment the filter switches on, and is stored in `my_face_shape.json`.
- Skin: blemishes and pimples removed automatically and tone evened out, while keeping fine skin detail so it stays clear (not blurry); softened nasolabial folds (法令纹)
- Makeup: learned from her video. `my_makeup.png` stores how her Douyin filter changes the color at every spot of her face (blush gradient, lip tint, aegyo-sal (卧蚕) highlight, brightening, soft shading), and the app maps it onto her live face.

**Accessories:** switch with **thumb + ring finger on your grab hand** (or press **E**):
none (the default) → Santa hat → reindeer antlers + red nose → Christmas tree hat → garland crown with bells → floating pastel stars → snow in hair

Everything follows your face as you move, and it sits under the fog like the rest of your reflection. The beauty settings are tuned for Shiqi's face (see the top of `facefx.py`).

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
- **Wrong camera (e.g. it uses your iPhone):** press **V** until you see yourself from the laptop camera. The app remembers that choice for next time. It also tries to pick the built-in FaceTime camera on its own the first time. To stop your Mac from ever using the iPhone as a webcam: on the iPhone, go to Settings → General → AirPlay & Continuity and turn off Continuity Camera.
- **`pip` can't find mediapipe:** your Python version is probably too new. Use Python 3.10–3.12, for example `python3.12 -m venv .venv`.
- **Fog triggers when you're just talking:** raise `MOUTH_OPEN` at the top of `main.py` (try `0.07`).
- **Want the fog to slowly come back after wiping:** set `FOG_REGROW_SECONDS = 25`.

## How it works

- **MediaPipe Face Landmarker** tracks your lips. When the gap between them is big enough, fog spreads out from your mouth.
- **MediaPipe Hand Landmarker** tracks 21 points on each hand. Fingertip distances detect pinches, and a closed fist wipes along the pinky edge of the hand (pinky knuckle to wrist). Hand size on screen (how close you are) controls sticker zoom.
- **Fog** is a mask over the view: heavily blurred, lightened, and tinted cool blue-white, glowing warm near the fire. Wiping and drawing cut clear lines into the mask.
- **Droplets** each work like a tiny lens. They show a flipped, sharper view of what's behind the glass, with a bright highlight, a dark rim, and light focused at the bottom. Some drops slide down and leave clear trails in the fog.
- **The glass** fills the screen with no frame. Frost grains creep in from the edges, granular snow is piled along the bottom, snow and soft fairy lights are outside, and firelight flickers and reflects in the glass.
- **The ornaments** are rendered with real lighting: a glossy glass sphere with window reflections and a brushed-gold cap, a frosted snowflake, a faceted gold star, and holly leaves with glossy berries.
- Everything is drawn with code (OpenCV + NumPy). There are no image files.

## Files

- `main.py`: camera loop, fog, gestures, stickers
- `window.py`: frost, snow, lighting and the world outside
- `droplets.py`: water droplets and sliding drops
- `facefx.py`: the face filters
- `accessories.py`: the watercolor-style Santa hat, reindeer antlers, tree hat and garland crown (painted once on first run, then cached)
- `my_face_shape.json`: Shiqi's measured Douyin face shape (reusable in future projects)
- `my_makeup.png` / `my_makeup.json`: Shiqi's learned Douyin makeup and skin tone
- `cut_stickers.py`: cuts a sticker sheet into separate transparent PNGs
- `stickers.py`: the ornaments
- `text.py`: elegant script fonts for the on-screen text
