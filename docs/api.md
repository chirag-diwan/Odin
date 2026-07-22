|API path|POST|GET|
|--|--|--|
|`/v1/chat/completions`|`Returns the infered tokens either via a server side event or via a single concatinated answer (depending on wether 'stream' was requested or not`|`NOT SUPPORTED`|

## JSON structure.
### Request
> [!WARNING]
> The `messages` field is set to be an array for compatibility BUT it should contain only ONE `system` , `user` pair because the engine keeps its own context , failing to do so will cause unexpected behavior
```jsonc
{
  "model": "your-custom-model-name", //Not supported will be ignored
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing."}
  ],
  "temperature": 0.7, //Not supported will be ignored
  "stream": false
}
```


### Invalid json error
```jsonc
{
  "error": {
    "message": "JSON parsing error , content field not set", //Message
    "type": "invalid_request_error" //Type
  }
}
```
### Response
#### Server side event (stream)
**The tokens will be streamed as soon as there is any**
```jsonc
data: {
  "object": "chat.completion.chunk",
  "choices": [{
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": tok
      },
      "finish_reason": "" //null always
  }]
}\n\n
```
**`data: [DONE]\n\n` is sent the the end of stream**



#### One shot
**The response will be given after the completion of the inference phase**

```jsonc
{
  "object": "chat.completion",
  "choices": [{
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Complete infered answer"
      },
      "finish_reason": "" //null , "stop" , "length" , "tool_callss"
  }],
  "usage": {
    "prompt_tokens": 14,
    "completion_tokens": 30,
    "total_tokens": 44
  }
}
```
