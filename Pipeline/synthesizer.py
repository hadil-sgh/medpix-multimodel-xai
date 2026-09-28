"""Box 6 - Synthesizer: turns (prediction + heatmap description + evidence) into one paragraph
that says WHAT was found, WHERE, and WHY. Two backends:
  * template : deterministic, no LLM (default; every sentence is traceable to a field)
  * phi3     : Phi-3-mini writes the paragraph, constrained to the evidence given in the prompt
"""
from . import config as C


def _fmt_words(ws):
    return ", ".join(f'"{w}"' for w, _ in ws) if ws else "no strongly indicative words"


def template_paragraph(case, fp, vision, evidence, path):
    s = [f"The image is predicted to be a {fp.modality} scan of the {fp.location.lower()} "
         f"(confidence {fp.p_location:.2f} for location, {fp.p_modality:.2f} for modality)."]
    s.append(f"Visual evidence: the model focused on a {vision['region']}, covering about {vision['area_pct']:.0f}% of the image.")
    s.append(f"Textual evidence: in the clinical history, the words {_fmt_words(evidence['text_words'])} support this body region.")
    if not fp.agree:
        s.append(f"Note: the image alone suggests {fp.image_location.lower()} while the history suggests "
                 f"{fp.text_location.lower()}; the final answer combines both.")
    cands = evidence.get("candidates", [])
    if cands:
        top = cands[0]
        uids = [n["uid"] for n in evidence["neighbours"] if n["diagnosis"] == top["diagnosis"]]
        s.append(f"Similar past cases suggest {top['diagnosis']} ({top['category'] or 'category unknown'}), "
                 f"seen in case(s) {', '.join(uids)}.")
        if top.get("kg_facts"):
            fx = "; ".join(f"{f['relation']} {f['object']}" for f in top["kg_facts"][:4])
            s.append(f"Knowledge graph: {top['diagnosis']} -> {fx}" +
                     (" (consistent with the body region seen in the image)." if top["kg_location_consistent"] else "."))
        if len(cands) > 1:
            s.append("Other possibilities: " + "; ".join(c["diagnosis"] for c in cands[1:]) + ".")
    if path == "short":
        s.append("This case took the short path (confident, image and history agree).")
    return " ".join(s)


class Synthesizer:
    def __init__(self, backend=C.SYNTH_BACKEND):
        self.backend = backend
        if backend == "phi3":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(C.LLM_ID)
            self.llm = AutoModelForCausalLM.from_pretrained(C.LLM_ID, dtype=torch.float16, device_map="auto")

    def __call__(self, case, fp, vision, evidence, path):
        draft = template_paragraph(case, fp, vision, evidence, path)
        if self.backend != "phi3":
            return draft
        facts = "\n".join(f"- {t['subject']} | {t['relation']} | {t['object']}" for t in evidence.get("kg", [])[:8])
        prompt = (
            "You are a radiology assistant. Rewrite the FINDINGS below as one clear paragraph for a clinician. "
            "Use ONLY the information given. Do not add new diagnoses, numbers or locations.\n\n"
            f"FINDINGS:\n{draft}\n\nKNOWLEDGE-GRAPH FACTS (optional context):\n{facts or '- none'}\n\nParagraph:")
        msgs = [{"role": "user", "content": prompt}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.llm.device)
        out = self.llm.generate(ids, max_new_tokens=220, do_sample=False)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
