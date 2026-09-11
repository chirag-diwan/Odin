#include "formatter.hpp"


void TemplateParamGenerator::SetDefault(Architecture arch) {
  std::string bos_token;

  if (arch == Architecture::LLAMA3) {
    bos_token = "<|begin_of_text|>";
  } else {
    bos_token = "<|im_start|>";
  }

  auto now = std::time(nullptr);
  auto* localTime = std::localtime(&now);
  std::ostringstream oss; oss << std::put_time(localTime, "%Y-%m-%d %H:%M:%S");

  current["bos_token"] = bos_token;
  current["date_string"] = oss.str();
  current["add_generation_prompt"] = true;
  current["tools_in_user_message"] = false;
}

void TemplateParamGenerator::SetTools(){
  current["tools"] = GetTools();
}

void TemplateParamGenerator::Reset() {
  current = {};
}

json& TemplateParamGenerator::GetRef() {
  return current;
}

void TemplateParamGenerator::AddMessage(const std::string& role, const std::string& content) {
  current["messages"].push_back(json{
    {"role", role},
      {"content", content},
  });
}

void TemplateParamGenerator::AddTool() {}

void Formatter::Init(const std::string& templ){
  format = minja::Parser::parse(templ, {}); 
}

std::string Formatter::GetFormattedString(const json& values) {
  return format->render(minja::Context::make(values));
}
