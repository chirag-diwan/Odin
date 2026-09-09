#pragma once

#include <memory>
#include <string>
#include <ctime>
#include <iomanip>

#include "../external/minja/minja.hpp"

#include "./types.hpp"


class TemplateParamGenerator {
  private:
    json current;

    static json GetTools() {
      return json::array({
        {
          {"type", "function"},
            {
              "function", {
                {"name", "python"},
                {
                  "description",
                  "Execute Python code and return the result."
                },
                {
                  "parameters", {
                    {"type", "object"},
                    {
                      "properties", {
                        {
                          "code", {
                            {"type", "string"},
                            {"description", "Python code to execute"}
                          }
                        }
                      }
                    },
                    {"required", {"code"}}
                  }
                }
              }
            }
        },
          {
            {"type", "function"},
            {
              "function", {
                {"name", "shell"},
                {
                  "description",
                  "Execute a shell command and return the result."
                },
                {
                  "parameters", {
                    {"type", "object"},
                    {
                      "properties", {
                        {
                          "command", {
                            {"type", "string"},
                            {"description", "Shell command to execute"}
                          }
                        }
                      }
                    },
                    {"required", {"command"}}
                  }
                }
              }
            }
          }
      });
    }

  public:
    void SetDefault(Architecture arch) {
      std::string bos_token;

      if (arch == Architecture::LLAMA3) {
        bos_token = "<|begin_of_text|>";
      } else {
        bos_token = "<|im_start|>";
      }

      auto now = std::time(nullptr);
      auto* localTime = std::localtime(&now);

      current["bos_token"] = bos_token;
      current["date_string"] =
        std::put_time(localTime, "%Y-%m-%d %H:%M:%S")._M_fmt;
      current["add_generation_prompt"] = true;
      current["tools_in_user_message"] = false;
      current["tools"] = GetTools();
    }

    static json GetDefault(const std::string& bos_token) {
      auto now = std::time(nullptr);
      auto* localTime = std::localtime(&now);

      json j;

      j["bos_token"] = bos_token;
      j["date_string"] =
        std::put_time(localTime, "%Y-%m-%d %H:%M:%S")._M_fmt;
      j["add_generation_prompt"] = true;
      j["tools_in_user_message"] = false;
      j["tools"] = GetTools();

      return j;
    }

    void Reset() {
      current = {};
    }

    json& GetRef() {
      return current;
    }

    void AddMessage(const std::string& role, const std::string& content) {
      current["messages"].push_back(json{
        {"role", role},
          {"content", content},
      });
    }

    void AddTool() {}
};


class Formatter {
  private:
    std::shared_ptr<minja::TemplateNode> format;

  public:
    void Init(const std::string& templ){
      format = minja::Parser::parse(templ, {}); 
    }

    std::string GetFormattedString(const json& values) {
      return format->render(minja::Context::make(values));
    }
};


//std::string GetFormatted(Architecture model_arch , const std::string& system , const std::string & user){
//  switch (model_arch) {
//    case Architecture::LLAMA3:
//      return std::format("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" , system , user);
//
//    case Architecture::QWEN2:
//      return std::format( "<|im_start|>system\n{}\n<|im_end|>\n" "<|im_start|>user\n{}\n<|im_end|>\n" "<|im_start|>assistant\n", system, user);
//    default:
//      Errorif(true, "Invalid model architecture");
//      return "";
//  }
//}
