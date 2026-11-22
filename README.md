# Interactive Design Lab

Interactive software experiments — computer vision, hand/body tracking (MediaPipe), and things you control with gestures. Some may connect to hardware later.

Each project lives in its own folder with:
- `README.md` — what it does + how to run it
- `requirements.txt` — the Python packages it needs

## Projects

| Project | What it does |
|---|---|
| [Christmas Fog Window](christmas-fog-window/) | Blow to fog a cozy Christmas window, wipe it with your hand, and decorate it with gesture-made stickers |

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
