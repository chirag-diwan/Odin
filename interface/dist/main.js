import { AssistantMessage, UserMessage } from "./elements.js";
import { SSEManager } from "./server-side-event.js";
async function main() {
    const history = document.getElementById("chat-history");
    const send_btn = document.getElementById("send-btn");
    const textarea = document.getElementById("prompt-area");
    const sse = new SSEManager("/stream");
    let turn = 'USER';
    let last_chat_bubble = null;
    const update_res = async () => {
        for await (const token of sse) {
            if (!last_chat_bubble) {
                last_chat_bubble = new AssistantMessage();
                history.appendChild(last_chat_bubble);
            }
            last_chat_bubble.textContent += token;
        }
        turn = "USER";
        last_chat_bubble = null;
    };
    const send_prompt = async (prompt) => {
        const res = await fetch('/prompt', {
            method: "POST",
            body: prompt
        });
        if (!res.ok) {
            alert('Error sending message');
        }
    };
    send_btn?.addEventListener('click', async () => {
        if (turn === 'USER') {
            const val = textarea.value;
            textarea.value = "";
            turn = 'SSE';
            history.appendChild(new UserMessage(val));
            await send_prompt(val);
            update_res();
        }
    });
}
main();
//# sourceMappingURL=main.js.map