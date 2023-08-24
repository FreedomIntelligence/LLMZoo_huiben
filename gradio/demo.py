import requests
import gradio as gr

def generate(story):
    picture_list, img_url_list = requests.get(f"/generate?story={story}")
    content_list = []
    for i in range(len(picture_list)):
        picture = picture_list[i]
        img_url = img_url_list[i]
        content = f"![Huiben Img]({img_url})\n{picture}"
        content_list.append(content)
    md = "\n".join(content_list)
    return md

def main():
    with gr.Blocks() as g:
        story = gr.Text(label="Input your story", lines=8)
        output = gr.Markdown("Output Here")
        btn = gr.Button("Submit")

        btn.click(generate, inputs=story, outputs=output)
    
    gr.close_all()
    g.launch(share=True)

if __name__ == "__main__":
    main()