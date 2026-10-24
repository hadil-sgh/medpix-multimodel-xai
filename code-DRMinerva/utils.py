from datasets import load_dataset, DatasetDict
from huggingface_hub import hf_hub_download
from PIL import Image
import re
import torch

def load_custom_dataset(split):
    path = 'MedPix-2-0/'
    if split == 'case-topic':
        file = 'case-topic.jsonl'
    elif split == 'descriptions':
        file = 'descriptions.jsonl'
    return load_dataset("json", data_files=path+file)

def load_complete_dataset(balanced=True):
    # balanced retrieves also train 1 and train 2 
    path = "MedPix-2-0/splitted_dataset/"
    if balanced:
        dataset_train_1 = load_dataset("json", data_files=path+"data_train_1.jsonl")
        dataset_train_2 = load_dataset("json", data_files=path+"data_train_2.jsonl")

    dataset_train = load_dataset("json", data_files=path+"data_train.jsonl")
    dataset_dev = load_dataset("json", data_files=path+"data_dev.jsonl")
    dataset_test = load_dataset("json", data_files=path+"data_test.jsonl")

    if balanced:
        dataset = DatasetDict({
            'train' : dataset_train['train'],
            'train_1' : dataset_train_1['train'],
            'train_2' : dataset_train_2['train'],
            'dev' : dataset_dev['train'],
            'test' : dataset_test['train'],
            })
    else:
        dataset = DatasetDict({
            'train' : dataset_train['train'],
            'dev' : dataset_dev['train'],
            'test' : dataset_test['train'],
            })

    return dataset

def load_complete_dataset_description(balanced=True):
    # balanced retrieves also train 1 and train 2 
    path = "MedPix-2-0/splitted_dataset/"
    if balanced:
        dataset_train_1 = load_dataset("json", data_files=path+"descriptions_train_1.jsonl")
        dataset_train_2 = load_dataset("json", data_files=path+"descriptions_train_2.jsonl")
    
    dataset_train = load_dataset("json", data_files=path+"descriptions_train.jsonl")
    dataset_dev = load_dataset("json", data_files=path+"descriptions_dev.jsonl")
    dataset_test = load_dataset("json", data_files=path+"descriptions_test.jsonl")

    if balanced:
        dataset = DatasetDict({
            'train' : dataset_train['train'],
            'train_1' : dataset_train_1['train'],
            'train_2' : dataset_train_2['train'],
            'dev' : dataset_dev['train'],
            'test' : dataset_test['train'],
            })
    else:
        dataset = DatasetDict({
            'train' : dataset_train['train'],
            'dev' : dataset_dev['train'],
            'test' : dataset_test['train'],
            })

    return dataset

def get_description_per_uid(dataset_description):
    description_per_uid = {}
    for split in ['train', 'dev', 'test']:
        for sample in dataset_description[split]:
            if sample['U_id'] not in description_per_uid:
                description_per_uid[sample['U_id']] = []
            description_per_uid[sample['U_id']].append(sample)
    return description_per_uid

def load_model(clip_vision_encoder_path, clip_vision_encoder_pretrained, lang_encoder_path, tokenizer_path, device, checkpoint=None):
    import gc
    import open_flamingo.src.factory as of_factory
    from transformers import AutoModelForCausalLM

    class _LM:
        @staticmethod
        def from_pretrained(*a, **k):
            k["trust_remote_code"] = True
            k["torch_dtype"] = torch.float16
            return AutoModelForCausalLM.from_pretrained(*a, **k)
    of_factory.AutoModelForCausalLM = _LM

    is_mpt = "mpt" in lang_encoder_path.lower()
    model, image_processor, tokenizer = of_factory.create_model_and_transforms(
        clip_vision_encoder_path=clip_vision_encoder_path,
        clip_vision_encoder_pretrained=clip_vision_encoder_pretrained,
        lang_encoder_path=lang_encoder_path,
        tokenizer_path=tokenizer_path,
        cross_attn_every_n_layers=1,
        decoder_layers_attr_name=None if is_mpt else "model.layers",
    )
    model = model.to(device, dtype=torch.float16)
    gc.collect()
    if checkpoint is not None:
        msd = {k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()}
        model.load_state_dict(msd, strict=False)
        print("Model loaded from checkpoint")
    else:
        ckpt = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
        sd = torch.load(ckpt, map_location="cpu", mmap=True)
        model.load_state_dict(sd, strict=False)
        del sd
    gc.collect(); torch.cuda.empty_cache()
    return model, image_processor, tokenizer

def load_model_cpu(clip_vision_encoder_path, clip_vision_encoder_pretrained, lang_encoder_path, tokenizer_path, checkpoint=None):
    # imported here so MedGemma runs do not need open_flamingo installed
    from open_flamingo import create_model_and_transforms
    model, image_processor, tokenizer = create_model_and_transforms(
        #device,
        clip_vision_encoder_path = clip_vision_encoder_path,
        clip_vision_encoder_pretrained = clip_vision_encoder_pretrained,
        lang_encoder_path = lang_encoder_path,
        tokenizer_path=tokenizer_path,
        cross_attn_every_n_layers=1,
        #cache_dir="PATH/TO/CACHE/DIR"  # Defaults to ~/.cache
    )

    if checkpoint is not None:
        msd = checkpoint["model_state_dict"]
        msd = {k.replace("module.", ""): v for k, v in msd.items()}
        model.load_state_dict(msd, strict=False)
        print("Model loaded from checkpoint")

    else:
        # grab model checkpoint from huggingface hub
        checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
        model.load_state_dict(torch.load(checkpoint_path), strict=False)

    return model, image_processor, tokenizer

def get_static_context(dataset, path, image_processor, description_per_uid, inference_label):
    # preparare visual context
    context_counter = 3
    sample_context = dataset['train'][0]
    history_context = get_block(sample_context, 'History', 'Case')

    descriptions_context = description_per_uid[sample_context['U_id']]
    vision_context = []
    context_str = ''
    
    for description in descriptions_context[:2]:
        image = Image.open(path+'images/'+description['image']+'.png')
        vision_context.append(image_processor(image).unsqueeze(0))
        # preparare textual context
        age_context = get_block(description, 'Age', 'Description')
        sex_context = get_block(description, 'Sex', 'Description')

        if inference_label == 'modality':
            context_str += f"<image> {age_context}{sex_context}patient. {history_context}\nThe image is a {description['Type']}.<|endofchunk|>"
        elif inference_label == 'location':
            context_str += f"<image> {age_context}{sex_context}patient. {history_context}\nThe image shows a {description['Location Category']}.<|endofchunk|>"
        elif inference_label == 'joint':
            context_str += f"<image> {age_context}{sex_context}patient. {history_context}\nThe image is a {description['Type']} scan showing a {description['Location Category']}.<|endofchunk|>"
    
    return context_str, vision_context, context_counter

def pick_sample(dataset, u_id):
    for sample_context in dataset['train_1']:
        if sample_context['U_id'] == u_id:
            return sample_context
        
def get_rag_context(sample, retrieved_reports, dataset, path, image_processor, description_per_uid, inference_label):
    context_counter = 1
    context_str = ''
    vision_context = []
    for context in retrieved_reports[sample['U_id']]:
        sample_context = pick_sample(dataset, context)
        descriptions_context = description_per_uid[sample_context['U_id']]
        history_context = get_block(sample_context, 'History', 'Case')
        
        for description in descriptions_context:
            age_context = get_block(description, 'Age', 'Description')
            sex_context = get_block(description, 'Sex', 'Description')
            image = Image.open(path+'images/'+description['image']+'.png')
            vision_context.append(image_processor(image).unsqueeze(0))
            if inference_label == 'modality':
                context_str += f"<image> {age_context}{sex_context}patient. {history_context}\nThe image is a {description['Type']}.<|endofchunk|>"
            elif inference_label == 'location':
                context_str += f"<image> {age_context}{sex_context}patient. {history_context}\nThe image shows a {description['Location Category']}.<|endofchunk|>"
            elif inference_label == 'joint':
                context_str += f"<image> {age_context}{sex_context}patient. {history_context}\nThe image is a {description['Type']} scan showing a {description['Location Category']}.<|endofchunk|>"
            context_counter += 1
    return context_str, vision_context, context_counter

# FT
def inference(visual_input, textual_input, model, tokenizer, device):
#def inference(visual_input, textual_input, model, tokenizer):
    vision_x = torch.cat(visual_input, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
    lang_x = tokenizer([textual_input], return_tensors="pt")

    generated_text = model.generate(
        vision_x=vision_x.to(device, dtype=torch.float16),
        lang_x=lang_x["input_ids"].to(device),
        attention_mask=lang_x["attention_mask"].to(device),
        max_new_tokens=20,
        num_beams=1)

    output = tokenizer.decode(generated_text[0])

    del vision_x
    del lang_x["input_ids"]
    del lang_x["attention_mask"]
    del generated_text

    torch.cuda.empty_cache()

    return output

# MedGemma-4b-it: replaces the Flamingo model, same RAG context (retrieved train_1 cases as few-shot examples)
MEDGEMMA_LOCS = ["Head", "Thorax", "Abdomen", "Spine and Muscles", "Reproductive and Urinary System"]
MEDGEMMA_SYN = {"brain": "Head", "skull": "Head", "chest": "Thorax", "lung": "Thorax",
                "spine": "Spine and Muscles", "pelvis": "Reproductive and Urinary System"}

def load_medgemma(model_id="google/medgemma-4b-it"):
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(model_id, quantization_config=bnb, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor

def medgemma_user_turn(image, age, sex, history, question):
    return {"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": f"{age}{sex}patient. {history}\n{question}"}]}

def get_rag_context_medgemma(sample, retrieved_reports, dataset, path, description_per_uid, question, max_images=4):
    # same retrieved cases as get_rag_context, as user/assistant chat turns; max_images=0 keeps every image
    turns = []
    for context in retrieved_reports[sample['U_id']]:
        sample_context = pick_sample(dataset, context)
        history_context = get_block(sample_context, 'History', 'Case')

        for description in description_per_uid[sample_context['U_id']]:
            if max_images and len(turns) // 2 >= max_images:
                return turns
            age_context = get_block(description, 'Age', 'Description')
            sex_context = get_block(description, 'Sex', 'Description')
            image = Image.open(path+'images/'+description['image']+'.png').convert("RGB")
            turns.append(medgemma_user_turn(image, age_context, sex_context, history_context, question))
            turns.append({"role": "assistant", "content": [{"type": "text", "text": f"Modality: {description['Type']}\nLocation: {description['Location Category']}"}]})
    return turns

def inference_medgemma(messages, model, processor, max_new_tokens=30):
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
    n = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    output = processor.decode(out[0][n:], skip_special_tokens=True).strip()

    del inputs
    del out
    torch.cuda.empty_cache()

    return output

def parse_medgemma(answer):
    # "Modality: MRI\nLocation: Brain" -> ("MR", "Head"); "?" when a field cannot be read
    m = re.search(r"modality:\s*(CT|MRI|MR)", answer, re.I)
    modality = ("MR" if m.group(1).upper().startswith("MR") else "CT") if m else "?"
    location = "?"
    l = re.search(r"location:\s*([^\n]+)", answer, re.I)
    if l:
        txt = l.group(1).strip().lower()
        for loc in MEDGEMMA_LOCS:
            if txt.startswith(loc.lower().split()[0]):
                location = loc
                break
        else:
            for k, v in MEDGEMMA_SYN.items():
                if k in txt:
                    location = v
                    break
    return modality, location

def get_block(sample, key, upper_key):
    # sex, treatment & follow up are the likely missing
    if key in sample[upper_key] and sample[upper_key][key] != "N/A" and sample[upper_key][key] is not None:
        if key == 'Age':
            return sample[upper_key][key] + ' y.o. '
        elif key == 'Location Category':
            return '('+sample[upper_key][key]+')'
        elif key == 'Treatment & Follow Up':
            return ' can be treated in this way: '+sample[upper_key][key]
        elif key == 'Caption':
            #return 'showing ' + sample[upper_key][key]
            return sample[upper_key][key]
        else:
            return sample[upper_key][key]+' '
    else:
        #print(f"Missing {key} in {sample['U_id']}")
        return ''
