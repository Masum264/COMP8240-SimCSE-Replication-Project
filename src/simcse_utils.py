"""Minimal SimCSE encoder using HuggingFace transformers only (no `simcse` pip package).

Pooling follows the authors' documented usage:
  supervised checkpoints   -> pooler_output          ("cls")
  unsupervised checkpoints -> last_hidden_state[:,0] ("cls_before_pooler")
"""


def default_pooler(model_name: str) -> str:
    return "cls_before_pooler" if "unsup" in model_name else "cls"


def load_encoder(model_name: str, device=None):
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return tok, model, device


def encode(texts, tok, model, device, pooler="cls", batch_size=64, max_length=128):
    """Return L2-normalised sentence embeddings as a (n, d) torch tensor."""
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[i:i + batch_size]]
            inputs = tok(batch, padding=True, truncation=True,
                         max_length=max_length, return_tensors="pt").to(device)
            res = model(**inputs, return_dict=True)
            if pooler == "cls":
                emb = res.pooler_output
            elif pooler == "cls_before_pooler":
                emb = res.last_hidden_state[:, 0]
            else:
                raise ValueError(f"unknown pooler {pooler!r}")
            out.append(torch.nn.functional.normalize(emb, dim=-1).cpu())
    return torch.cat(out)


def pair_cosine(a, b, tok, model, device, pooler="cls", **kw):
    """Cosine similarity for each (a[i], b[i]) pair, as a numpy array."""
    ea = encode(list(a), tok, model, device, pooler, **kw)
    eb = encode(list(b), tok, model, device, pooler, **kw)
    return (ea * eb).sum(-1).numpy()
