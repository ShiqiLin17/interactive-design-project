# Interactive Design Lab

Interactive software experiments — computer vision, hand/body tracking (MediaPipe), and things you control with gestures. Some may connect to hardware later.

Each project lives in its own folder with:
- `README.md` — what it does + how to run it
- `requirements.txt` — the Python packages it needs

## Projects

| Project | What it does |
|---|---|
| [Air Throw](air-throw/) | Grab stickers, photos and your profile card in the air and throw them to friends' phones |
| [Personal Color Test](color-palette/) | Korean-style personal color drape test with finger gestures and a side-by-side photo comparison |
| [Christmas Fog Window](christmas-fog-window/) | Realistic fogged winter window: blow to fog, wipe or write with your fingers, decorate with gesture stickers |

## Setup (once)

```bash
git clone https://github.com/ShiqiLin17/interactive-design-project.git
cd interactive-design-project
```

## Get the latest code

```bash
git pull
```

## Run a project

```bash
cd <project-folder>
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
