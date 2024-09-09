# Quick Start

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```sh
python main.py
python model.py

fastapi run --workers 4 main.py
uvicorn main:app --host 0.0.0.0 --port 8080 --workers 4
```

## Documentation

```sh
pdoc --output-dir docs model.py
```
