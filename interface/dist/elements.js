export class ChatMessage extends HTMLElement {
    constructor(type) {
        super();
        this.row = document.createElement("div");
        this.row.className = `message-row ${type}`;
        this.bubble = document.createElement("div");
        this.bubble.className = "message-bubble";
        this.row.appendChild(this.bubble);
        this.appendChild(this.row);
    }
    update(content) {
        this.bubble.textContent = content;
    }
    updateHTML(html) {
        this.bubble.innerHTML = html;
    }
    clear() {
        this.bubble.textContent = "";
    }
}
export class UserMessage extends ChatMessage {
    constructor(text = "") {
        super("user");
        if (text)
            this.update(text);
    }
}
export class AssistantMessage extends ChatMessage {
    constructor(text = "") {
        super("assistant");
        if (text)
            this.update(text);
    }
}
customElements.define("assistant-message", AssistantMessage);
customElements.define("user-message", UserMessage);
//# sourceMappingURL=elements.js.map