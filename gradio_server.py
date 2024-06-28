import gradio as gr
import random
from recurrentgpt import RecurrentGPT
from human_simulator import Human
from sentence_transformers import SentenceTransformer
from utils import get_init, parse_instructions
from starlette.requests import Request
import re


_CACHE = {}

# Build the semantic search model
embedder = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
def formated_user_story(title, outline, p1, p2, p3):
     return f"""
Title: {title}
Outline: {outline}
Paragraph 1: {p1}
Paragraph 2: {p2}
Paragraph 3: {p3}
"""

def init_prompt(novel_type, description, language, written_paras):
    if description != "":
        description = "about " + description

    if language != "":
        description += f" in {language} language"
    
    if written_paras != "":
        description += f"based on the provided content <{written_paras}>"

    return f"""
Please write a <{novel_type}> {description} with 50 chapters. Follow the format below precisely:

Begin with the title of the novel.
Next, write an outline for the first chapter. The outline should describe the background and the beginning of the novel.
Write the first three paragraphs with their indication of the novel based on your outline. Write in a novelistic style and take your time to set the scene.
Write a summary that captures the key information of the three paragraphs.
Finally, write three different instructions for what to write next, each containing around five sentences. Each instruction should present a possible, interesting continuation of the story.
The output format should follow these guidelines:
Title: <title of the novel>
Outline: <outline for the first chapter>
Paragraph 1: <content for paragraph 1>
Paragraph 2: <content for paragraph 2>
Paragraph 3: <content for paragraph 3>
Summary: <content of summary>
Instruction 1: <content for instruction 1>
Instruction 2: <content for instruction 2>
Instruction 3: <content for instruction 3>

Make sure to be precise and follow the output format strictly.

"""

def init_prompt_continue(novel_type, description, language, formated_user_story):

    if description == "":
        description = ""
    else:
        description = "about " + description
    if language != "":
        description += f" in {language} language"
    return f"""
Here is the user provided content:
{formated_user_story}

Write a novel in the style of <{novel_type}> {description} based on the provided content. 
Follow the format below precisely:
First a summary that captures the key information of the three paragraphs.
Then write three different instructions for what to write next, each containing around five sentences. Each instruction should present a possible, interesting continuation of the story.
The output format should follow these guidelines:

Summary: <content of summary>
Instruction 1: <content for instruction 1>
Instruction 2: <content for instruction 2>
Instruction 3: <content for instruction 3>

Make sure to be precise and follow the output format strictly. Do not exceed the word count or modify the content in any way.

"""

def init(novel_type, description, language, written_paras, save_story, request: gr.Request):
    out_file = None
    if save_story == "Yes" or save_story == "是":
        out_file = f"{novel_type}_{description}_{language}.txt"

    cookie = request.headers.get('cookie', None)
    if cookie is None:
        cookie = ""
    else:
        cookie = request.headers['cookie']
        cookie = cookie.split('; _gat_gtag')[0]

    if novel_type == "":
        # novel_type = "Science Fiction"
        novel_type = "叙事"

    global _CACHE
    init_paragraphs = "" 
    init_paragraphs = get_init(text=init_prompt(novel_type, description, language, written_paras),response_file=out_file)
    
    start_input_to_human = {
        'output_paragraph': "",
        'input_paragraph': '\n\n'.join([init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2'], init_paragraphs['Paragraph 3']]),
        'output_memory': init_paragraphs['Summary'],
        'output_instruction': [init_paragraphs['Instruction 1'], init_paragraphs['Instruction 2'], init_paragraphs['Instruction 3']]
    }

    _CACHE[cookie] = {"start_input_to_human": start_input_to_human,
                      "init_paragraphs": init_paragraphs}
    written_paras = f"""Title: {init_paragraphs['Title']}

Outline: {init_paragraphs['Outline']}

Paragraphs:

{start_input_to_human['input_paragraph']}"""
    long_memory = parse_instructions([init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2'], init_paragraphs['Paragraph 3']])
    # short memory, long memory, current written paragraphs, 3 next instructions
    # print("inital", written_paras)
    return start_input_to_human['output_memory'], long_memory, written_paras, init_paragraphs['Instruction 1'], init_paragraphs['Instruction 2'], init_paragraphs['Instruction 3']

def init_continue(novel_type, description, language, title, outline, p1, p2, p3, written_paras, save_story, request: gr.Request):
    out_file = None
    if save_story == "Yes" or save_story == "是":
        out_file = f"{novel_type}_{description}_{language}.txt"

    cookie = request.headers.get('cookie', None)
    if cookie is None:
        cookie = ""
    else:
        cookie = request.headers['cookie']
        cookie = cookie.split('; _gat_gtag')[0]

    if novel_type == "":
        novel_type = "叙事"

    global _CACHE
    init_paragraphs = ""
    formated_story_start = formated_user_story(title, outline, p1, p2, p3)

    init_paragraphs = get_init(init_text=formated_story_start, text=init_prompt_continue(novel_type, description, language, formated_story_start),response_file=out_file)

    start_input_to_human = {
        'output_paragraph': "",
        'input_paragraph': '\n\n'.join([init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2'], init_paragraphs['Paragraph 3']]),
        'output_memory': init_paragraphs['Summary'],
        'output_instruction': [init_paragraphs['Instruction 1'], init_paragraphs['Instruction 2'], init_paragraphs['Instruction 3']]
    }

    _CACHE[cookie] = {"start_input_to_human": start_input_to_human,
                      "init_paragraphs": init_paragraphs}
    written_paras = f"""Title: {init_paragraphs['Title']}

Outline: {init_paragraphs['Outline']}

Paragraphs:

{start_input_to_human['input_paragraph']}"""
    long_memory = parse_instructions([init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2'], init_paragraphs['Paragraph 3']])
    # short memory, long memory, current written paragraphs, 3 next instructions
    # print("inital", written_paras)
    return start_input_to_human['output_memory'], long_memory, written_paras, init_paragraphs['Instruction 1'], init_paragraphs['Instruction 2'], init_paragraphs['Instruction 3']

def step(novel_type, description, language, short_memory, long_memory, save_story, instruction1, instruction2, instruction3, current_paras, request: gr.Request,):
    out_file = None
    if save_story == "Yes":
        out_file = f"{novel_type}_{description}_{language}.txt"
    
    if current_paras == "":
        return "", "", "", "", "", ""
    global _CACHE
    print("step request", request)
    cookie = request.headers.get('cookie', None)
    if cookie is None:
        # Handle the case where the cookie is not present
        cookie = ""
    else:
        cookie = request.headers['cookie']
        cookie = cookie.split('; _gat_gtag')[0]
    cache = _CACHE[cookie]

    if "writer" not in cache:
        start_input_to_human = cache["start_input_to_human"]
        start_input_to_human['output_instruction'] = [
            instruction1, instruction2, instruction3]
        init_paragraphs = cache["init_paragraphs"]
        human = Human(input=start_input_to_human,
                      memory=None, embedder=embedder, language=language, output_file=out_file)
        human.step()
        start_short_memory = init_paragraphs['Summary']
        writer_start_input = human.output

        # Init writerGPT
        writer = RecurrentGPT(input=writer_start_input, short_memory=start_short_memory, long_memory=[
            init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2']], memory_index=None, embedder=embedder, language=language, output_file=out_file)
        cache["writer"] = writer
        cache["human"] = human
        writer.step()
    else:
        human = cache["human"]
        writer = cache["writer"]
        output = writer.output
        output['output_memory'] = short_memory
        #randomly select one instruction out of three
        instruction_index = random.randint(0,2)
        output['output_instruction'] = [instruction1, instruction2, instruction3][instruction_index]
        human.input = output
        human.step()
        writer.input = human.output
        writer.step()

    long_memory = parse_instructions(writer.long_memory)
    # short memory, long memory, current written paragraphs, 3 next instructions
    return writer.output['output_memory'], long_memory, current_paras + '\n\n' + writer.output['input_paragraph'], human.output['output_instruction'], *writer.output['output_instruction']

def controled_step(novel_type, description, language, short_memory, save_story, long_memory, selected_instruction, current_paras, request: gr.Request, ):
    out_file = None
    if save_story == "Yes" or save_story == "是":
        out_file = f"{novel_type}_{description}_{language}.txt"
    
    if current_paras == "":
        return "", "", "", "", "", ""
    global _CACHE
    # print("control step request", request.headers)
    cookie = request.headers.get('cookie', None)
    if cookie is None:
        # Handle the case where the cookie is not present
        cookie = ""
    else:
        cookie = request.headers['cookie']
        cookie = cookie.split('; _gat_gtag')[0]
    cache = _CACHE[cookie]
    # print("cache", cache)

    if "writer" not in cache:
        start_input_to_human = cache["start_input_to_human"]
        start_input_to_human['output_instruction'] = selected_instruction
        init_paragraphs = cache["init_paragraphs"]
        human = Human(input=start_input_to_human,
                      memory=None, embedder=embedder, language=language, output_file=out_file)
        human.step()
        start_short_memory = init_paragraphs['Summary']
        writer_start_input = human.output

        # Init writerGPT
        writer = RecurrentGPT(input=writer_start_input, short_memory=start_short_memory, long_memory=[
            init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2'], init_paragraphs['Paragraph 3']], memory_index=None, embedder=embedder, language=language, output_file=out_file)
        cache["writer"] = writer
        cache["human"] = human
        writer.step()
    else:
        human = cache["human"]
        writer = cache["writer"]
        output = writer.output
        output['output_memory'] = short_memory
        output['output_instruction'] = selected_instruction
        human.input = output
        human.step()
        writer.input = human.output
        writer.step()
    long_memory = parse_instructions(writer.long_memory)
    # short memory, long memory, current written paragraphs, 3 next instructions
    return writer.output['output_memory'], long_memory, current_paras + '\n\n' + writer.output['input_paragraph'], selected_instruction, *writer.output['output_instruction']

# SelectData is a subclass of EventData
def on_select(instruction1, instruction2, instruction3, evt: gr.SelectData):
    selected_plan = int(evt.value.replace("情节发展 ", ""))
    # selected_plan = int(evt.value.replace("Instruction ", ""))
    selected_plan = [instruction1, instruction2, instruction3][selected_plan-1]
    return selected_plan

with gr.Blocks(title="RecurrentGPT", css="footer {visibility: hidden}", theme="default") as demo:
    # new tab
    with gr.Tab("选择性故事生成"):
        with gr.Column():
            with gr.Row():
                novel_type = gr.Textbox(
                    label="小说类型", placeholder="例如: 幻想架空")
                description = gr.Textbox(label="主题")
                # language = gr.Textbox(label="Language")
                language = gr.Radio(choices=["English", "中文"], label="语言")
                save_story = gr.Radio(choices=["是", "否"], label="保存故事")
            gr.Examples(["科幻", "言情", "悬疑", "架空",
                        "历史", "恐怖", "搞笑", "传记"],
                        inputs=[novel_type], elem_id="example_selector")
            btn_init = gr.Button(
                "生成故事", elem_id="init_button")
            written_paras = gr.Textbox(
                label="故事主体段落", lines=21)
                

        with gr.Column():
            gr.Markdown("### 记忆模块")
            short_memory = gr.Textbox(
                label="短期记忆", lines=3)
            long_memory = gr.Textbox(
                label="长期记忆", lines=6)
            gr.Markdown("### 故事情节发展模块")
            instruction1 = gr.Textbox(
                label="情节发展 1 (可修改)", lines=4)
            instruction2 = gr.Textbox(
                label="情节发展 2 (可修改)", lines=4)
            instruction3 = gr.Textbox(
                label="情节发展 3 (可修改)", lines=4)
            last_step = gr.Textbox(
                label="上一个选择的情节", lines=2)
        with gr.Column():
            with gr.Column(scale=1, min_width=100):
                            selected_plan = gr.Radio(["情节发展 1", "情节发展  2", "情节发展 3"], label="选择发展")
            with gr.Column(scale=3, min_width=300):
                            selected_instruction = gr.Textbox(
                                label="选择的情节发展（可修改）", max_lines=5, lines=5)

        btn_step = gr.Button("下一步", elem_id="step_button")
        btn_init.click(init, inputs=[novel_type, description, language, written_paras, save_story], outputs=[
            short_memory, long_memory, written_paras, instruction1, instruction2, instruction3])
        btn_step.click(controled_step, inputs=[novel_type, description, language, save_story, short_memory, long_memory, selected_instruction, written_paras], outputs=[
            short_memory, long_memory, written_paras, last_step, instruction1, instruction2, instruction3])
        selected_plan.select(on_select, inputs=[
                             instruction1, instruction2, instruction3], outputs=[selected_instruction])
    
    # new tab
    with gr.Tab("故事续写"):
        with gr.Column():
            with gr.Row():
                novel_type = gr.Textbox(
                    label="小说类型", placeholder="例如: 幻想架空")
                description = gr.Textbox(label="主题")
                # language = gr.Textbox(label="Language")
                language = gr.Radio(choices=["English", "中文"], label="语言")
                save_story = gr.Radio(choices=["是", "否"], label="保存故事")
            gr.Examples(["科幻", "言情", "悬疑", "架空",
                        "历史", "恐怖", "搞笑", "传记"],
                        inputs=[novel_type], elem_id="example_selector")
            with gr.Row():
                title = gr.Textbox(
                    label="小说标题", placeholder="例如: 大喵喵厨师探险")
                outline = gr.Textbox(label="主要情节", placeholder="几句话，简单概括故事设定背景和主要人物信息")
            paragraph1 = gr.Textbox(label="段落一", placeholder="段落一 内容 (不超过五句话)")
            paragraph2 = gr.Textbox(label="段落二", placeholder="段落二 内容 (不超过五句话)")
            paragraph3 = gr.Textbox(label="段落三", placeholder="段落三 内容 (不超过五句话)")
            save_story = gr.Radio(choices=["是", "否"], label="保存故事")

            btn_init = gr.Button(
                "续写故事", elem_id="init_button")
            written_paras = gr.Textbox(
                label="故事主体段落", lines=21)
                

        with gr.Column():
            gr.Markdown("### 记忆模块")
            short_memory = gr.Textbox(
                label="短期记忆", lines=3)
            long_memory = gr.Textbox(
                label="长期记忆", lines=6)
            gr.Markdown("### 故事情节发展模块")
            instruction1 = gr.Textbox(
                label="情节发展 1 (可修改)", lines=4)
            instruction2 = gr.Textbox(
                label="情节发展 2 (可修改)", lines=4)
            instruction3 = gr.Textbox(
                label="情节发展 3 (可修改)", lines=4)
            last_step = gr.Textbox(
                label="上一个选择的情节", lines=2)
        with gr.Column():
            with gr.Column(scale=1, min_width=100):
                            selected_plan = gr.Radio(["情节发展 1", "情节发展  2", "情节发展 3"], label="选择发展")
            with gr.Column(scale=3, min_width=300):
                            selected_instruction = gr.Textbox(
                                label="选择的情节发展（可修改）", max_lines=5, lines=5)

        btn_step = gr.Button("下一步", elem_id="step_button")
        btn_init.click(init_continue, inputs=[novel_type, description, language, title, outline, paragraph1, paragraph2, paragraph3, written_paras, save_story], outputs=[
            short_memory, long_memory, written_paras, instruction1, instruction2, instruction3])
        btn_step.click(controled_step, inputs=[novel_type, description, language, save_story, short_memory, long_memory, selected_instruction, written_paras], outputs=[
            short_memory, long_memory, written_paras, last_step, instruction1, instruction2, instruction3])
        selected_plan.select(on_select, inputs=[
                             instruction1, instruction2, instruction3], outputs=[selected_instruction])


    demo.queue(max_size=20)
    demo.launch(max_threads=1, inbrowser=True, share=True)

if __name__ == "__main__":
    demo.launch(server_port=8005, share=True,
                server_name="0.0.0.0", show_api=False)