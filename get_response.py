from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

def get_response(inputs,outputs,tokenizer,num_return):
    responses_list=[]
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
    gen_kwargs = {'num_return_sequences': 1, 'min_new_tokens': 10 ,'max_length':2048, 'num_beams':2,
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
    gen_kwargs = {'num_return_sequences': 1, 'min_new_tokens': 10 ,'max_length':2048, 'num_beams':2,
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
    