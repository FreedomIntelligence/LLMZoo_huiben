from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import torch
import time

def get_response(inputs,outputs,tokenizer,num_return):
    responses_list=[]
    # for output in outputs:
    # responses = [tokenizer.decode(output,skip_special_tokens=True) for output in outputs]
    batch_return=[]
    for i, output in enumerate(outputs):
        input_len = len(inputs[0])
        generated_output = output[input_len:]
        batch_return.append(tokenizer.decode(generated_output, skip_special_tokens=True))
        if i%num_return==num_return-1:
            responses_list.append(batch_return)
            batch_return=[]
    return responses_list

def load_and_generate(human,model_path="/workspace2/junzhi/LLMZoo/phoenix_7b_junzhi"):
    accelerator = Accelerator()
    gen_kwargs = {'num_return_sequences': 1, 'min_new_tokens': 10 ,'max_length':1024, 'num_beams':2,
            'do_sample':True, 'top_p':1.0, 'temperature':1.0, 'repetition_penalty':1.5}
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,padding_side='left')
    # if args.process_tokenizer:
    #     tokenizer.pad_token = tokenizer.unk_token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half()
    except Exception as e:
        print(e)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, ignore_mismatched_sizes=True).half()
    model = model.eval()
    system = "A chat between a curious human and an artificial intelligence assistant. \nThe assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
    prompt = f"""{system}<Human>: {human} <Assistant>: """
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    # print(tokenizer.decode(input_ids))
    model,input_ids = accelerator.prepare(model,input_ids)
    input_ids = input_ids.to('cuda')
    # print(model.device)
    # print(input_ids.device)
    outputs = accelerator.unwrap_model(model).generate(input_ids,**gen_kwargs)
    response = get_response(input_ids,outputs,tokenizer,1)
    return response
    # outputs = model(**inputs)
    # print(tokenizer.decode(outputs))
    #/workspace2/liangjuhao/conda_envs/phoenix

    # from transformers import pipeline
    # generator = pipeline(model="/workspace2/junzhi/LLMZoo/phoenix_7b_junzhi")
    # print(generator("Hello world!"))

# def txt_to_img(prompt, seed):
#     pipeline = DiffusionPipeline.from_pretrained("/workspace2/junzhi/dreamlike_anime")
#     generator = torch.Generator("cuda").manual_seed(seed)
#     pipeline.to("cuda")
#     # negative_prompt = """
#     # broken hand, unnatural body, simple background, duplicate, retro style, low quality, lowest quality, bad anatomy,
#     # bad proportions, extra digits, duplicate, watermark, signature, text, extra digit, fewer digits, worst quality,
#     # jpeg artifacts, blurry
#     # """
#     # info = """
#     # Steps: 35, Sampler: Euler a, CFG scale: 7, Seed: 1217462402, Face restoration: GFPGAN, Size: 768x1024, Model hash: cae1bee30e, Model: illuminatiDiffusionV1_v11"
#     # """
#     image = pipeline(prompt=prompt, generator=generator).images[0]
#     return image
#     # image.save("image.png")

def get_model(model_path="/workspace2/junzhi/LLMZoo/phoenix_7b_junzhi"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,padding_side='left')
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half()
    except Exception as e:
        print(e)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, ignore_mismatched_sizes=True).half()
    return model, tokenizer

def generate(human, model, tokenizer):
    accelerator = Accelerator()
    gen_kwargs = {'num_return_sequences': 1, 'min_new_tokens': 10 ,'max_length':1024, 'num_beams':2,
            'do_sample':True, 'top_p':1.0, 'temperature':1.0, 'repetition_penalty':1.0}
    model = model.eval()
    system = "A chat between a curious human and an artificial intelligence assistant. \nThe assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
    prompt = f"""{system}<Human>: {human} <Assistant>: """
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    model,input_ids = accelerator.prepare(model,input_ids)
    input_ids = input_ids.to('cuda')
    outputs = accelerator.unwrap_model(model).generate(input_ids,**gen_kwargs)
    response = get_response(input_ids,outputs,tokenizer,1)
    return response[0][0]

# if __name__ == "__main__":
#     torch.cuda.empty_cache()
#     human = """
#     Generate a detailed prompt from this sentence: red apple
#     """
#     after = """
#     Generate a detailed prompt from this sentence: dinasour fighting with each other
#     """
#     model, tokenizer = get_model(model_path="/workspace2/junzhi/checkpoints/prompt_create")
#     s1 = time.time()
#     p1 = generate(human, model, tokenizer)
#     e1 = time.time()
#     s2 = time.time()
#     p2 = generate(after, model, tokenizer)
#     e2 = time.time()
#     print(p1)
#     print(p2)
#     print(f"g1 duration {e1-s1}")
#     print(f"g2 duration {e2-s2}")

    # prompt = load_and_generate(human, model_path="/workspace2/junzhi/checkpoints/prompt_create")
    # p2 = load_and_generate(human, model_path="/workspace2/junzhi/checkpoints/prompt_create")
    # print(prompt[0][0])
    # print(p2[0][0])
    