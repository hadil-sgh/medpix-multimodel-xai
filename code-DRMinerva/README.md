# DR-Minerva
DR-Minerva: a Multimodal Language Model based on Minerva for Diagnostic Information Retrieval

Code for training and evaluation of DR-Minerva 

File in this folder, listed according to their execution order:

- `RAG_construction.py` for building of the RAG module, as for template and vectorstore creation 
- `finetune-minerva.py` for fine-tuneìing textual encoder and generate lora adpters
- `merge_weights.py` for saving fine-tuned model and merges adapters
- `inference_rag_gen.py` for retrieving from vectorstore for subsequent inference
- `gen_dataset_ft_flamingo` for arranging dataset for Flamingo fine-tuning
- `train.py` for training for flamingo architecture
- `eval.py` for inference with flamingo + fine-tuned textual encoder
- `evaluate-code.py` for evaluate the obtained output in termes of accuracy
- `utils.py` for utility functions

To run inference with MedGemma-4b-it (4-bit, fits a Colab T4) instead of Flamingo, keeping the same RAG context:
`python eval.py --llm_model medgemma --rag --inference_split test --n_exp 7` (needs `pip install -U transformers accelerate bitsandbytes` and a Hugging Face login with access to `google/medgemma-4b-it`).
`--max_ctx_images` caps the retrieved example images per prompt (default 4, 0 = all), `--n_cases` runs only the first N cases.

Please cite our work as follows:

```
@inproceedings{drMinerva,
    author = {Siragusa, Irene and Contino, Salvatore and Pirrone, Roberto},
    title = {{DR-Minerva: a Multimodal Language Model based on Minerva for Diagnostic Information Retrieval}},
    booktitle="AIxIA 2024 -- Advances in Artificial Intelligence",
    year="2025",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="288--300",
    isbn="978-3-031-80607-0"
}
```
