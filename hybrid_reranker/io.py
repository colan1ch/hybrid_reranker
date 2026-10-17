import json

try:
    import fcntl
except ImportError:  # Windows: no advisory locks, run one API command at a time there
    fcntl = None


def read_jsonl(path):
    if path.exists():
        with path.open() as file:
            return [json.loads(line) for line in file if line.strip()]
    return []


def load_results(path):
    return {row["id"]: row for row in read_jsonl(path)}


def append_jsonl(path, row):
    """Append one line; the lock keeps lines whole when two commands share a file such as data/api.jsonl."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        if fcntl:
            fcntl.flock(file, fcntl.LOCK_EX)
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
