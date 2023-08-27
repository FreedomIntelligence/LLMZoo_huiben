import requests
import gradio as gr

def generate(story):
    server_address = "http://localhost:8068"
    response = requests.get(f"{server_address}/generate", params={"story": story})
    if response.status_code == 200:
        res = response.json()
        # print(res)
        picture_list = res["picture_list"]
        img_url_list = res["img_url_list"]
        content_list = []
        for i in range(len(picture_list)):
            picture = picture_list[i]
            img_url = img_url_list[i]
            content = f"![Huiben Img]({img_url})\n{picture}"
            content_list.append(content)
        md = "\n".join(content_list)
        return md
    else:
        return response.status_code
    

def main():
    with gr.Blocks() as g:
        story = gr.Text(label="Input your story", lines=8)
        output = gr.Markdown("## Output Here")
        btn = gr.Button("Submit")

        btn.click(generate, inputs=story, outputs=output)
    
    gr.close_all()
    g.launch(server_name="0.0.0.0", server_port=7860, share=True, max_threads=200)

if __name__ == "__main__":
    main()