#include <regex>
#include <string_view>
#include <vector>
class Tokeniser{
  private:
    std::vector<std::string_view> tokens;
    std::regex pre_tokeniser;
  public:
    Tokeniser(){
    }
};
