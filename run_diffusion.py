from diffusers import DiffusionPipeline
import torch
class image_generator():
    def __init__(self) -> None:
        self.pipeline = DiffusionPipeline.from_pretrained("/workspace2/junzhi/dreamlike_anime")
        self.pipeline.to("cuda")

    def generate_img(self, prompt, negative_prompt, generator):
        image = self.pipeline(prompt=prompt, negative_prompt=negative_prompt, generator=generator).images[0]
        return image

# pipeline = DiffusionPipeline.from_pretrained("/workspace2/junzhi/dreamlike_anime")
# seed = 1217462402
# generator = torch.Generator("cuda").manual_seed(seed)
# pipeline.to("cuda")
# prompt = """
# a still of amy is pushing a small patient to take an x-ray. she is careful to keep the patient safe, correct composition, award winning photo, anime style
# """
# negative_prompt = """
# broken hand, unnatural body, simple background, duplicate, retro style, low quality, lowest quality, bad anatomy,
# bad proportions, extra digits, duplicate, watermark, signature, text, extra digit, fewer digits, worst quality,
# jpeg artifacts, blurry
# """
# image = pipeline(prompt=prompt).images[0]
# image.save("amy.png")