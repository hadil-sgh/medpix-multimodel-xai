import argparse, os, pickle, re, time
import pandas as pd
import torch
from llama_index.core import KnowledgeGraphIndex, StorageContext, Settings, Document, PromptTemplate
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

RELATIONS = ["located in", "imaged with", "presents with", "treated with", "is a", "diagnosed by"]

SCHEMA_PROMPT = PromptTemplate(
    "Extract up to {max_knowledge_triplets} knowledge triplets from the medical text below.\n"
    "Rules:\n"
    "- The SUBJECT must always be the disease name.\n"
    "- Use ONLY these relations: " + ", ".join(RELATIONS) + ".\n"
    "- Keep each object short (at most 6 words).\n"
    "- Ignore case IDs, ages and sexes. Skip a relation if the text does not state it.\n"
    "- Output one triplet per line as (subject, relation, object) and nothing else.\n"
    "---------------------\n"
    "Text: Clinical report of a 45 female patient suffering from a Meningioma displayed in MR - T1 weighted. "
    "Headaches for 3 months. The disease Meningioma located in Brain, Meninges (Head) and it can be treated "
    "in this way: surgical resection. About Meningioma we can say that: a benign tumour of the meninges, "
    "diagnosed by histology.\n"
    "Triplets:\n"
    "(Meningioma, imaged with, MR)\n"
    "(Meningioma, presents with, headaches)\n"
    "(Meningioma, located in, Brain meninges)\n"
    "(Meningioma, located in, Head)\n"
    "(Meningioma, treated with, surgical resection)\n"
    "(Meningioma, is a, benign tumour)\n"
    "(Meningioma, diagnosed by, histology)\n"
    "---------------------\n"
    "Text: {text}\n"
    "Triplets:\n"
)


def phi3_completion_to_prompt(c):
    return f"<|user|>\n{c}<|end|>\n<|assistant|>\n"


def phi3_messages_to_prompt(messages):
    out = ""
    for m in messages:
        role = m.role.value if hasattr(m.role, "value") else str(m.role)
        role = role if role in ("system", "assistant") else "user"
        out += f"<|{role}|>\n{m.content}<|end|>\n"
    return out + "<|assistant|>\n"


def strip_case_id(text):
    # "MPX1106 is a clinical report of ..." -> "Clinical report of ..."  (avoids 'Mpx1106 -> Is -> Clinical report')
    return re.sub(r"^\s*MPX\d+\s+is\s+a\s+clinical\s+report\s+of", "Clinical report of", text, flags=re.I)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hf_model", default="microsoft/Phi-3-mini-4k-instruct")
    p.add_argument("--llm_model", default=None, help="short name in output paths (default: phi3mini / phi3schema)")
    p.add_argument("--embed_model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--split", default="train")
    p.add_argument("--relations", type=int, default=10, help="max triplets per chunk")
    p.add_argument("--typology", default="hybrid")
    p.add_argument("--limit", type=int, default=0, help="only the first N documents (0 = all)")
    p.add_argument("--dedup", action="store_true", help="one document per case (U_id) instead of per image")
    p.add_argument("--schema", action="store_true", help="disease-centred graph with fixed relations")
    p.add_argument("--lower", action="store_true")
    a = p.parse_args()
    llm_name = a.llm_model or ("phi3schema" if a.schema else "phi3mini")

    path = "MedPix-2-0/"
    df = pd.read_csv(f"{path}KG/templates/template-{a.split}.csv")
    if a.dedup:
        df = df.drop_duplicates("U_id").reset_index(drop=True)
    if a.limit:
        df = df.head(a.limit)
    print(f"{len(df)} documents from template-{a.split}.csv | mode: {'schema' if a.schema else 'free-form'}")

    docs = []
    for i in range(len(df)):
        t, u = str(df.at[i, "Description"]), str(df.at[i, "U_id"])
        t = strip_case_id(t)
        if a.lower:
            t, u = t.lower(), u.lower()
        docs.append(Document(text=t, metadata={"U_id": u},
                             excluded_llm_metadata_keys=["U_id"], excluded_embed_metadata_keys=["U_id"]))

    print("loading LLM", a.hf_model)
    Settings.llm = HuggingFaceLLM(
        model_name=a.hf_model, tokenizer_name=a.hf_model,
        context_window=4096, max_new_tokens=256,
        model_kwargs={"torch_dtype": torch.float16},
        generate_kwargs={"do_sample": False},
        completion_to_prompt=phi3_completion_to_prompt,
        messages_to_prompt=phi3_messages_to_prompt,
        stopping_ids=[32000, 32007],  # <|endoftext|>, <|end|>
        device_map="auto",
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name=a.embed_model)
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 50

    storage_context = StorageContext.from_defaults(graph_store=SimpleGraphStore())
    kwargs = dict(documents=docs, max_triplets_per_chunk=a.relations, storage_context=storage_context,
                  include_embeddings=True, show_progress=True)
    if a.schema:
        kwargs["kg_triple_extract_template"] = SCHEMA_PROMPT
    t0 = time.time()
    index = KnowledgeGraphIndex.from_documents(**kwargs)
    print(f"KG built in {(time.time() - t0) / 60:.1f} min")

    low = "-lower" if a.lower else ""
    name = f"{a.typology}-{a.split}-{a.relations}trips{llm_name}{low}" + (f"-limit{a.limit}" if a.limit else "")
    out_dir = f"{path}KG/graphs/{name}"
    os.makedirs(out_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=out_dir)
    print("saved index ->", out_dir)

    g = index.get_networkx_graph()
    print(f"graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    with open(f"{path}KG/graphs-{name}.pickle", "wb") as f:
        pickle.dump(g, f)
    try:
        from pyvis.network import Network
        net = Network(notebook=False, cdn_resources="in_line", directed=True)
        net.from_nx(g)
        net.write_html(f"{path}KG/graphs-{name}.html")
        print("saved interactive graph ->", f"{path}KG/graphs-{name}.html")
    except Exception as e:
        print("pyvis export skipped:", e)

    # quality check: how many facts use the allowed relations?
    gd = storage_context.graph_store._data.graph_dict
    triplets = [(s, r, o) for s, rels in gd.items() for r, o in rels]
    on_schema = sum(r.strip().lower() in RELATIONS for _, r, _ in triplets)
    print(f"facts: {len(triplets)} | using the 6 allowed relations: {on_schema} "
          f"({100 * on_schema / max(len(triplets), 1):.0f}%)")
    for s, r, o in triplets[:15]:
        print(f"   {s}  ->  {r}  ->  {o}")