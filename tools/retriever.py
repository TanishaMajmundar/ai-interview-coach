import hashlib, json
from pathlib import Path
import chromadb

KB_PATH = Path("data/resources.json")
DB_DIR = "data/chroma"
_col = None


def _get_collection():
    """Open Chroma and (re)index the knowledge base when resources.json changed."""
    global _col
    if _col is not None:
        return _col
    raw = KB_PATH.read_bytes()
    kb_hash = hashlib.md5(raw).hexdigest()
    items = json.loads(raw.decode("utf-8"))
    client = chromadb.PersistentClient(path=DB_DIR)
    meta = {"hnsw:space": "cosine", "kb_hash": kb_hash}
    col = client.get_or_create_collection("resources", metadata=meta)
    if col.count() == 0 or (col.metadata or {}).get("kb_hash") != kb_hash:
        client.delete_collection("resources")
        col = client.get_or_create_collection("resources", metadata=meta)
        col.add(
            ids=[i["id"] for i in items],
            documents=[f'{i["topic"]}. {i["title"]}. {i["note"]} {" ".join(i.get("tags", []))}' for i in items],
            metadatas=[{k: i[k] for k in ("title", "link", "note", "topic")} for i in items],
        )
    _col = col
    return col


def retrieve_resources(query: str, k: int = 3, max_distance: float = 0.75) -> list:
    """Return up to k knowledge-base records close to the query. Links come from the KB only."""
    col = _get_collection()
    res = col.query(query_texts=[query], n_results=min(k, col.count()))
    out = []
    for rid, meta, dist in zip(res["ids"][0], res["metadatas"][0], res["distances"][0]):
        if dist <= max_distance:
            out.append({"id": rid, **meta, "distance": round(dist, 3)})
    return out