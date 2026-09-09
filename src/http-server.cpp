#include "../include/model_utils.hpp"
#include "../include/engine.hpp"
#include "../include/json_tokeniser.hpp"
#include "../include/ggufparser.hpp"
#include "../include/config.hpp"
#include "../include/logging.hpp"
#include "../include/types.hpp"
#include "../include/formatter.hpp"
#include "../include/http-manager.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "main-utility.hpp"
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <sys/mman.h>

static std::sig_atomic_t interupt = false;

void sig_int_handler(int){
  interupt = true;
}

void abort_callback(const char * message){
  Log(ERROR , "failed with" , message);
}

int main(int argc, char** argv) {
  ggml_set_abort_callback(abort_callback);

  std::signal(SIGINT , sig_int_handler);

  if (argc < 2) {
    return EXIT_FAILURE;
  }

  Config config = ParseConfig(argc, argv);
  GGufParser parser(config.modelPath);

  auto [addr, len] = parser.GetParsedFile();
  MmapGuard mmap_guard(addr, len); 

  ggml_backend_t backend = ggml_backend_cpu_init();
  auto threadpool_params = ggml_threadpool_params_default(std::thread::hardware_concurrency());

  UniqueThreadpool threadpool(ggml_threadpool_new(&threadpool_params));

  ggml_backend_cpu_set_threadpool(backend, threadpool.get());

  ggml_init_params static_ctx_params = {
    .mem_size = 10 * 1024 * 1024,
    .mem_buffer = NULL,
    .no_alloc = true 
  };

  UniqueGgmlContext static_ctx(ggml_init(static_ctx_params));

  auto model = CreateModel(static_ctx.get(), parser);
  if(model.globals.generalModelArchitecture == Architecture::UNKNOWN){
    //TODO Try and get more information about the Architecture using the full name.
    Log(ERROR , "Unknown model architecture" , model.globals.fullArchitectureName);
    return -1;
  }

  Engine engine(model, static_ctx.get(), backend);
  engine.ReservePrefillMemory();
  engine.ReserveDecodeMemory();

  BPETokeniser tokeniser(config.tokeniserJsonPath);
  std::vector<uint32_t> tokens;

  TemplateParamGenerator tpgenerator; tpgenerator.SetDefault(model.globals.generalModelArchitecture);
  Formatter formatter{std::string{model.globals.chat_template}};

  HttpManager manager(interupt);
  manager.StartListen();

  std::string system_prompt; system_prompt.reserve(32);

  std::string raw_prompt;

  while (!interupt) {
    tpgenerator.Reset();
    tpgenerator.SetDefault(model.globals.generalModelArchitecture);

    auto prompt_req = manager.ReadPrompt();
    raw_prompt = prompt_req.content;

    if(prompt_req.role != Role::USER){
      if(system_prompt.length() >= 8192) system_prompt.clear();
      system_prompt.append(raw_prompt);
      continue;
    }

    if (raw_prompt.empty()) {
      continue;
    }

    if(system_prompt.empty()){
      system_prompt = "You are a help full AI agent.";
    }

    if(raw_prompt.starts_with("!clear-context")){
      engine.ClearContext();
    }

    tpgenerator.AddMessage("system", system_prompt);
    tpgenerator.AddMessage("user", raw_prompt);
    auto prompt = formatter.GetFormattedString(tpgenerator.GetRef());

    size_t last_index = tokens.size();
    tokeniser.Tokenise(prompt, tokens);

    size_t span_size = tokens.size() - last_index;
    std::span<uint32_t> tokens_view(tokens.data() + last_index, span_size);
    manager.SetPromptTokens(span_size);

    uint32_t next_token = engine.Prefill(tokens_view);
    tokens.push_back(next_token);
    auto tok = tokeniser.Decode(next_token);
    if(tok.has_value()){
      manager.WriteInfered(*tok);
    }

    while (!interupt && (next_token != model.globals.ggmlEosTokenId)) {
      next_token = engine.Infer(tokens.back());
      tokens.push_back(next_token);

      if (next_token != model.globals.ggmlEosTokenId) {
        auto tok = tokeniser.Decode(next_token);
        if(tok.has_value()){
          manager.WriteInfered(*tok);
        }
      }
    }

    manager.WriteInfered(manager.DONE_TOK);

    interupt = false;
  }

  manager.stop();

  return EXIT_SUCCESS;
}
