export class SSEManager {
    constructor(url) {
        this.queue = [];
        this.waiting = [];
        this.closed = false;
        this.eventSource = new EventSource(url);
        this.eventSource.onmessage = (event) => {
            if (this.waiting.length > 0) {
                this.waiting.shift()({ value: event.data, done: false });
            }
            else {
                this.queue.push(event.data);
            }
        };
        this.eventSource.onerror = () => {
            this.closed = true;
            this.eventSource.close();
            while (this.waiting.length > 0) {
                this.waiting.shift()({ value: undefined, done: true });
            }
        };
    }
    async *[Symbol.asyncIterator]() {
        while (!this.closed) {
            if (this.queue.length > 0) {
                yield this.queue.shift();
            }
            else {
                const result = await new Promise(resolve => this.waiting.push(resolve));
                if (result.done)
                    return;
                yield result.value;
            }
        }
    }
}
//# sourceMappingURL=server-side-event.js.map