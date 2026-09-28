"""Box 7 - Verifier: checks the explanation before it is released.
  1. consistency  - the paragraph names the predicted body region and modality
  2. grounding    - every diagnosis the paragraph mentions exists in the gathered evidence
  3. faithfulness - hiding the Grad-CAM region hurts the prediction more than hiding random pixels
  4. confidence   - location confidence above the routing threshold
Returns pass/fail + the list of failed checks; the pipeline uses it to send a case back once.
"""
from . import config as C


def verify(paragraph, fp, vision, evidence, all_known_diagnoses):
    text = paragraph.lower()
    checks = {}
    checks["consistency"] = fp.location.lower() in text and fp.modality.lower() in text
    allowed = {c["diagnosis"].lower() for c in evidence.get("candidates", [])}
    mentioned = {d for d in all_known_diagnoses if d and len(d) > 4 and d in text}
    # a mention is supported if it is (part of) one of the candidate diagnoses taken from the evidence
    unsupported = sorted(d for d in mentioned if not any(d in a or a in d for a in allowed))
    checks["grounding"] = not unsupported
    checks["faithfulness"] = bool(vision["deletion"]["faithful"])
    checks["confidence"] = fp.p_location >= C.ROUTE_CONF_THRESHOLD
    failed = [k for k, ok in checks.items() if not ok]
    return {"passed": not failed, "checks": checks, "failed": failed, "unsupported_diagnoses": unsupported}
