import asyncio
import time
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from living_harness.memory.harness_context import HarnessContext
from living_harness.core.tools import HarnessTools
import re

app = FastAPI(title="Living Harness")

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Living Harness Chat</title>
        <style>
            body { font-family: sans-serif; margin: 20px; }
            #chat { height: 60vh; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; background: #f9f9f9; }
            .msg { margin-bottom: 5px; }
            .user { color: blue; }
            .sys { color: green; font-style: italic; }
            .ai { color: #333; }
        </style>
    </head>
    <body>
        <h1>Living Harness Chat</h1>
        <div id="chat"></div>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off" style="width: 80%; padding: 5px;"/>
            <button style="padding: 5px;">Send</button>
        </form>
        <script>
            var ws = new WebSocket("ws://localhost:8000/ws");
            var chat = document.getElementById('chat');
            ws.onmessage = function(event) {
                var p = document.createElement('div');
                p.className = 'msg ai';

                if (event.data.startsWith("[SYSTEM]")) {
                    p.className = 'msg sys';
                }

                p.innerText = event.data;
                chat.appendChild(p);
                chat.scrollTop = chat.scrollHeight;
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText");
                ws.send(input.value);

                var p = document.createElement('div');
                p.className = 'msg user';
                p.innerHTML = "<b>You:</b> " + input.value;
                chat.appendChild(p);
                chat.scrollTop = chat.scrollHeight;

                input.value = '';
                event.preventDefault();
            }
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

def load_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    model_name = "Qwen/Qwen3.5-0.8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model

model_loaded = False
tokenizer = None
model = None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global model_loaded, tokenizer, model
    await websocket.accept()

    if not model_loaded:
        await websocket.send_text("[SYSTEM]: Loading model Qwen/Qwen3.5-0.8B, please wait...")
        tokenizer, model = load_model()
        model_loaded = True
        await websocket.send_text("[SYSTEM]: Model loaded!")

    ctx = HarnessContext(N=4096, initial_t=50)
    ctx.system_prompt = """Ты - живая нейронная сеть.
У тебя есть внутренний монолог, который не видит пользователь. Он находится между тегами <think> и </think>.
Ты можешь использовать инструменты, написав их в монологе:
- ask_user(text): задать вопрос пользователю (например: ask_user("Привет!"))
- new_memory(key, val): сохранить память
- overwrite_memory(key, val): перезаписать память
Пользователь не видит твои размышления, поэтому если хочешь ему что-то сказать, обязательно используй ask_user!"""
    ctx.n1 = len(tokenizer.encode(ctx.system_prompt))

    tools = HarnessTools(ctx, websocket=websocket)

    await websocket.send_text("[SYSTEM]: Waiting for first message to animate model...")
    first_msg = await websocket.receive_text()

    ctx.sliding_window.append(f"<|im_start|>user\n{first_msg}<|im_end|>\n<|im_start|>assistant\n<think>\n")

    user_inputs = []

    async def listen_for_user():
        while True:
            try:
                data = await websocket.receive_text()
                user_inputs.append(data)
            except Exception:
                break

    asyncio.create_task(listen_for_user())

    unparsed_buffer = ""
    warning_triggered = False

    try:
        while True:
            if user_inputs:
                msg = user_inputs.pop(0)
                ctx.sliding_window.append(f"\n</think>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n<think>\n")
                await websocket.send_text("[SYSTEM]: Interrupted thought process with user message.")

            current_window_text = "".join(ctx.sliding_window)
            window_tokens = len(tokenizer.encode(current_window_text))

            keep_tokens, warning = ctx.check_and_truncate_window(window_tokens)

            if warning and not warning_triggered:
                 ctx.sliding_window.append(warning)
                 await websocket.send_text("[SYSTEM]: 85% Warning injected.")
                 warning_triggered = True

            if keep_tokens != window_tokens:
                 tokens = tokenizer.encode(current_window_text)
                 truncated_text = tokenizer.decode(tokens[-keep_tokens:])
                 ctx.sliding_window = [truncated_text]
                 await websocket.send_text("[SYSTEM]: Sliding window truncated (95% -> 10%).")
                 warning_triggered = False

            full_prompt = ctx.build_full_context()
            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[1]

            t0 = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.8
            )
            t1 = time.time()

            generated_ids = outputs[0][input_len:]
            new_generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

            ctx.add_generation_timestamp(t1, len(generated_ids))

            unparsed_buffer += new_generated_text

            if "ask_user(" in unparsed_buffer:
                match = re.search(r'ask_user\("(.*?)"\)', unparsed_buffer)
                if match:
                    await tools.ask_user(match.group(1))
                    unparsed_buffer = unparsed_buffer[match.end():]

            if "new_memory(" in unparsed_buffer:
                match = re.search(r'new_memory\("(.*?)",\s*"(.*?)"\)', unparsed_buffer)
                if match:
                    res = tools.new_memory(match.group(1), match.group(2))
                    await websocket.send_text(f"[SYSTEM-MEMORY]: {res}")
                    unparsed_buffer = unparsed_buffer[match.end():]

            if "overwrite_memory(" in unparsed_buffer:
                 match = re.search(r'overwrite_memory\("(.*?)",\s*"(.*?)"\)', unparsed_buffer)
                 if match:
                     res = tools.overwrite_memory(match.group(1), match.group(2))
                     await websocket.send_text(f"[SYSTEM-MEMORY]: {res}")
                     unparsed_buffer = unparsed_buffer[match.end():]

            if len(unparsed_buffer) > 500:
                unparsed_buffer = unparsed_buffer[-250:]

            print(f"[THOUGHT]: {new_generated_text}")

            if new_generated_text.endswith("<|im_end|>") or new_generated_text.endswith("<|endoftext|>"):
                new_generated_text = new_generated_text.replace("<|im_end|>", "")
                new_generated_text = new_generated_text.replace("<|endoftext|>", "")
                new_generated_text += "\n<think>\n"
                await websocket.send_text("[SYSTEM]: Forced into think mode.")

            elif "</think>" in new_generated_text:
                new_generated_text = new_generated_text.replace("</think>", "\nкстати, ")
                await websocket.send_text("[SYSTEM]: Prevented thinking exit.")

            ctx.sliding_window.append(new_generated_text)

            await asyncio.sleep(0.05)

    except Exception as e:
        print(f"Client disconnected or error: {e}")

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
