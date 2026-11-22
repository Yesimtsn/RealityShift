"""Text of claim versions.

The facts of a version are fixed by its edit (claims.apply_edit); a text
generator only puts them into words. Two backends:
  TemplateText  - deterministic, no AI (default for tests and tonight's runs)
  OllamaText    - local LLM (qwen2.5:7b, model id 845dbda0ea48) via Ollama
Every generated text is stored in a record (JSON). In replay mode the record
is the only source, so a replayed run is identical even though the LLM is not
deterministic (brief Q2: a seed alone does not guarantee identical LLM output).
"""
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Optional

from .claims import FACTS

STYLE = {
    "urgent": {"en": "BREAKING: ", "tr": "SON DAKİKA: "},
    "alarm": {"en": " Share before it is deleted!", "tr": " Silinmeden paylaşın!"},
}


def fact_texts(facts, lang):
    return [FACTS[f].en if lang == "en" else FACTS[f].tr for f in facts]


class TemplateText:
    model_id = "template-v1"

    def generate(self, prompt: dict) -> str:
        lang = prompt["lang"]
        body = "; ".join(prompt["facts"])
        body = body[0].upper() + body[1:] + "."
        pre = STYLE["urgent"][lang] if "urgent" in prompt["style"] else ""
        post = STYLE["alarm"][lang] if "alarm" in prompt["style"] else ""
        return pre + body + post


class OllamaText:
    """Calls a local Ollama server. Not used by the tests."""

    def __init__(self, model: str = "qwen2.5:7b", model_digest: str = "845dbda0ea48",
                 temperature: float = 0.7, url: str = "http://localhost:11434/api/generate"):
        self.model = model
        self.model_id = f"{model}@{model_digest}"
        self.temperature = temperature
        self.url = url

    def generate(self, prompt: dict) -> str:
        lang_name = "Turkish" if prompt["lang"] == "tr" else "English"
        instruction = (
            f"Write one short social-media post in {lang_name}. "
            f"It must state exactly these facts and nothing else: {prompt['facts']}. "
            f"Tone: {', '.join(prompt['style']) or 'neutral'}. "
            "Do not add any new numbers, names, places or sources. Output only the post."
        )
        body = json.dumps({"model": self.model, "prompt": instruction, "stream": False,
                           "options": {"temperature": self.temperature}}).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["response"].strip()


class TextGen:
    """Wraps a backend with a record. mode: 'record' (generate if missing) or 'replay' (record only)."""

    def __init__(self, backend=None, record_path: Optional[str] = None, mode: str = "record"):
        self.backend = backend or TemplateText()
        self.path = Path(record_path) if record_path else None
        self.mode = mode
        self.record = {}
        if self.path and self.path.exists():
            self.record = json.loads(self.path.read_text(encoding="utf-8"))

    def key(self, prompt: dict, call_key: str) -> str:
        payload = json.dumps({"prompt": prompt, "model": self.backend.model_id, "call": call_key},
                             sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def write(self, parent_text, edit, facts, style, lang, call_key: str) -> str:
        prompt = {"parent": parent_text, "edit": edit, "facts": fact_texts(facts, lang),
                  "style": list(style), "lang": lang}
        k = self.key(prompt, call_key)
        if k not in self.record:
            if self.mode == "replay":
                raise KeyError(f"replay: no recorded text for {call_key}")
            self.record[k] = self.backend.generate(prompt)
        return self.record[k]

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.record, ensure_ascii=False, indent=1), encoding="utf-8")
