from run_diffusion import image_generator
from fastapi import FastAPI
import uvicorn
import torch
import random
import os
import uuid
import socket
from typing import List


def get_local_ip():
    try:
        # 创建一个UDP套接字
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # 连接一个临时的目标地址和端口
        sock.connect(("8.8.8.8", 80))
        
        # 获取本地IP地址
        ip_address = sock.getsockname()[0]
        
        # 关闭套接字连接
        sock.close()
        
        return ip_address
    except socket.error:
        return "无法获取本机IP"


# 设置上传文件保存的目录
UPLOAD_DIRECTORY = "/workspace2/junzhi/LLMZoo_huiben/huiben"  # 替换为您希望保存上传文件的目录
local_ip = get_local_ip() # 调用函数获取本机IP地址
local_host = '0.0.0.0'
port = 8070
app_port = 8065

app = FastAPI()

@app.get("/print_huiben")
async def huiben_print(prompt: str, seed: int):
    prompt = prompt.split("<sep>")
    print(prompt)
    negative_prompt = ["""
    broken hand, unnatural body, simple background, duplicate, naked, nude
    """]*len(prompt)
    generator = torch.Generator("cuda").manual_seed(seed)
    imgs = g.generate_img(prompt, negative_prompt, generator)
    img_url = []
    for img in imgs:
        filename = f"{uuid.uuid4()}.png"
        img.save(os.path.join(UPLOAD_DIRECTORY, filename))
        file_url = f"http://{local_ip}:{app_port}/image/{filename}"  # 替换为您的域名和实际的图片路径
        img_url.append(file_url)
    return {
        "img_url": img_url
    }
    # result = 
    # print(result)
    # return result

if __name__ == "__main__":
    g = image_generator()
    uvicorn.run(app, host=local_host, port=port)