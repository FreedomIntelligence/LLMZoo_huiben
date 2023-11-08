import requests
import gradio as gr

def generate(story, page_num):
    server_address = "http://localhost:8065"
    response = requests.get(f"{server_address}/generate", params={"story": story, "page_num": page_num})
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
        page_num = gr.Text(label="Input your desired page number", lines=1)
        output = gr.Markdown("## Output Here")
        btn = gr.Button("Submit")

        btn.click(generate, inputs=[story, page_num], outputs=output)
    
    gr.close_all()
    g.launch(server_name="0.0.0.0", server_port=7861, share=True)

if __name__ == "__main__":
    main()