# -*- coding: utf-8 -*-
from get_response import get_model, generate
from run_diffusion import image_generator
import random
import torch
import re
import os
#加上用英文输出的测验
def get_frame(input, model, tokenizer):
    huiben_frame = generate(input, model, tokenizer)
    # print(huiben_frame)
    huiben_frame = huiben_frame.replace(":", "：")
    # print(huiben_frame)
    picture_list = split_picture(huiben_frame)
    return picture_list
    
def split_picture(huiben_frame):
    picture_list = re.split(r'画面\d+：', huiben_frame)
    picture_list = [b.strip() for b in picture_list if b.strip()]
    return picture_list

def get_pos_after_keyword(key_word, string):
    return string.find(key_word)+len(key_word), len(key_word)

def get_one_des(picture, model, tokenizer):
    huiben_des = generate(picture, model, tokenizer)
    des = []

    p1, l1 = get_pos_after_keyword("Subject: ", huiben_des)
    p2, l2 = get_pos_after_keyword("Detail: ", huiben_des)
    p3, l3 = get_pos_after_keyword("Environment: ", huiben_des)
    p4, l4 = get_pos_after_keyword("Color: ", huiben_des)
    # des.append(huiben_des[p1:p2-l2-1].strip())
    des.append(huiben_des[p2:p3-l3-1].strip())
    des.append(huiben_des[p3:p4-l4-1].strip())
    des.append(huiben_des[p4:].strip())
    huiben_des = ", ".join(des)
    # print(huiben_des)
    return huiben_des


#def get_prompt()

def generate_md_output():
    return None

def generate_huiben(story):
    model, tokenizer = get_model(model_path="/workspace2/junzhi/checkpoints/huiben")

    seed = random.randint(1, 10000000000)

    huiben_frame_prompt = """
    你接下来将扮演一个资深的绘本改编者，给你一个故事，你可以输出这个故事改编成的绘本的画面数以及每个画面的描述。

    输出格式：
    画面1:xxx\n  画面2:xxx\n ...... (xxx是具体的内容)。

    注意：
    请尽量使得每一个画面的内容不太一样
    请确保你的改编完全基于输入的故事，不要自己添加无关的情节和内容。
    尽可能多的转述出原输入的故事，保留输入故事中对话的内容
    请使用规定的输出格式，即：
    画面1:xxx\n  画面2:xxx\n ...... (xxx是具体的内容)。
    不要输出任何无关的输出

    输入：


    """

    huiben_des_prompt = """
    Next, you will take on the role of a seasoned picture book illustrator. Given a textual description of one page from a picture book story, you need to create a visual depiction of that page's illustration.

    Your answer should include descriptions of the visual details for the illustration page:

    Subject: The main content of the illustration. It can be characters, animals, or scenery. There can be multiple subjects.
    Detail: Detailed description of the actions, attire, expressions, etc., of the subjects in the illustration. Each subject needs to be described.
    Environment: Indoor, outdoor, underwater, space, forest, on an airplane, etc.
    Color: Vibrant, muted, pastel, bright, etc.

    Output Format:

    Subject:
    Detail:
    Environment:
    Color:

    Note:

    1. Illustration creation should be based on the original description from the picture book. Avoid adding irrelevant details.
    2. The output should follow the given format strictly. Do not provide any extraneous output or omit any part. The output format is:
    Subject:
    Detail:
    Environment:
    Color:

    3. Avoid using pronouns in the illustration description. Instead, use specific subject names for reference.
    4. The description of the illustration should focus solely on the depiction of the visual scene, omitting psychological activities and reflections unrelated to the image.
    5. In each part of the output format, provide your answer as short as possible but preserve key infomation, please use concise language. Make sure the whole sentence is less than 77 words
    6. Please answer the question using English

    Input: 


    """
    img_list = []
    prompts = []
    # story = input()
    # story = """
    #     在很久很久以前，有一只聪明的猴子住在一个山林里。一天晚上，月亮升起时，照耀在清澈的池塘里，猴子看到月亮的倒影在水面上，认为那是一颗可口的水果，于是他渴望捞到月亮吃。
    #     于是，猴子开始用手捞水，试图抓住月亮。但不管他怎么努力，月亮的倒影总是随着水面的波动而不断移动，根本无法捞到。猴子越是急躁，月亮的倒影看起来就越是迷人。他拼命地伸手捞取，却只是空中抓了几下。
    #     最终，猴子放弃了捞月的念头，他明白自己永远也无法抓到水中的月亮。但他也从中得到了教训：不能被虚幻的东西迷惑，要学会理智和冷静，珍惜眼前的现实。
    # """
    frame_input = huiben_frame_prompt+story
    # print(frame_input)
    picture_list = get_frame(frame_input, model, tokenizer)
    # print(picture_list)
    for picture in picture_list:
        des_input = huiben_des_prompt+"画面："+picture
        # print(des_input)
        huiben_des = get_one_des(des_input, model, tokenizer)
        prompts.append(huiben_des + ", anime style, 4k, highly detailed")
    # for prompt in prompts:
    #     print(len(prompt))
    #     print(prompt)
    torch.cuda.empty_cache()
    model, tokenizer = None, None
    img_generator = image_generator()
    for i, prompt in enumerate(prompts):
        negative_prompt = """
        broken hand, unnatural body, simple background, duplicate, retro style, low quality, lowest quality, bad anatomy,
        bad proportions, extra digits, duplicate, watermark, signature, text, extra digit, fewer digits, worst quality,
        jpeg artifacts, blurry, naked, nude
        """
        generator = torch.Generator("cuda").manual_seed(seed)
        img = img_generator.generate_img(prompt, negative_prompt, generator)
        img_list.append(img)
        # img.save(f"huiben/{i}.png")
    return picture_list, img_list