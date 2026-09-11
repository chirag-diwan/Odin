#pragma once

#include "nlohmann/json.hpp"
#include "minja/minja.hpp"
#include <memory>
#include <sstream>
#include <string>
#include <ctime>
#include <iomanip>


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
        }
      });
    }

  public:
    void SetDefault(Architecture arch) ;

    static json GetDefault(const std::string& bos_token) {
      auto now = std::time(nullptr);
      auto* localTime = std::localtime(&now);

      json j;
      std::ostringstream oss; oss << std::put_time(localTime, "%Y-%m-%d %H:%M:%S");

      j["bos_token"] = bos_token;
      j["date_string"] = oss.str();
      j["add_generation_prompt"] = true;
      j["tools_in_user_message"] = false;

      return j;
    }

    void SetTools();

    void Reset() ;

    json& GetRef() ;

    void AddMessage(const std::string& role, const std::string& content) ;

    void AddTool();
};


class Formatter {
  private:
    std::shared_ptr<minja::TemplateNode> format;

  public:
    void Init(const std::string& templ);

    std::string GetFormattedString(const json& values);
};
