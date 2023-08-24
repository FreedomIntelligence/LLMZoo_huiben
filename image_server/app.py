from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil, uuid
import os, glob
from starlette.responses import HTMLResponse
import uvicorn
import socket
from diffusers import DiffusionPipeline
import torch
import argparse
from generate_huiben import generate_huiben

app = FastAPI()
# model_path = "/workspace2/junzhi/dreamlike_anime"
model_path = "/workspace2/liangjuhao/models/dreamlik-photoreal-2.0"

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
local_host = '0.0.0.0'
local_ip = get_local_ip() # 调用函数获取本机IP地址
port = 8068
debug = False

@app.get("/image/{filename}")
async def get_image(filename: str):
    image_path = os.path.join(UPLOAD_DIRECTORY, filename)  # 图片文件的路径

    return FileResponse(image_path, media_type="image/jpeg")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    #上传file然后把file的url返回给客户端
    # image_data: data = open("filepath.jpg", 'rb')
    # 将上传的文件保存到指定目录
    if file.filename == "file":
        file.filename = f"{uuid.uuid4()}.jpg"
    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # 生成可访问的URL
    file_url = f"http://{local_ip}:{port}/image/{file.filename}"  # 替换为您的域名和实际的图片路径

    return file_url

@app.get("/generate")
async def get_huiben(story:str):
    if debug:
        return "http://tmp_url.png"
    img_url_list = []
    picture_list, img_list = generate_huiben(story)
    for image in img_list:
        filename = f"{uuid.uuid4()}.png"
        image.save(os.path.join(UPLOAD_DIRECTORY, filename))
        file_url = f"http://{local_ip}:{port}/image/{filename}"  # 替换为您的域名和实际的图片路径
        img_url_list.append(file_url)
    return picture_list, img_url_list

@app.get("/list")
async def show_images():
    filepaths = glob.glob(f"{UPLOAD_DIRECTORY}/*.jpg")
    filenames = [os.path.basename(f) for f in filepaths]
    return filenames

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running image server")
    parser.add_argument("--debug", type=bool, default=False, help="Whether or not actually call the API.")

    args = parser.parse_args()
    debug = args.debug
    uvicorn.run(app, host=local_host, port=port)