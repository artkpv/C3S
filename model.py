import torch
from transformers import LlamaForCausalLM, AutoTokenizer
import logging

def create_model(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    hf_model = LlamaForCausalLM.from_pretrained(
        cfg.model.model_name,
        torch_dtype=cfg.model.dtype if cfg.model.dtype == 'auto' else getattr(torch, cfg.model.dtype),
        device_map=cfg._device,
        low_cpu_mem_usage=True,
        load_in_8bit=cfg.model.load_in_8bit,
    )

    hf_model.eval()
    logging.info(hf_model)
    logging.info(hf_model.config)
    with torch.no_grad():
        logging.info(
            tokenizer.batch_decode(
                hf_model.generate(
                    tokenizer(
                        "The capital of Russia is", return_tensors="pt"
                    ).input_ids.to(cfg._device),
                    max_length=20,
                )
            )[0]
        )
    return hf_model, tokenizer
