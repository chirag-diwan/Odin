#include "./Encoder.h"

#include <cstdint>
#include <sys/types.h>

#define LIMITERCODE 2048

namespace Odin {
const std::string LIMITER_STR = "Ġ";

void BPEEncoder(const std::string&                         prompt,
                std::unordered_map<uint32_t, std::string>& vocab,
                std::vector<uint32_t>&                     tokens) {
  tokens.clear();
  for (const char c : prompt) {
    if (c == ' ') {
      vocab[LIMITERCODE] = LIMITER_STR;
      tokens.push_back(LIMITERCODE);
    } else {
      uint32_t token = static_cast<uint8_t>(c);
      vocab[token]   = std::string(1, c);
      tokens.push_back(token);
    }
  }

  for (uint8_t pass = 0; pass < 16; pass++) {
    if (tokens.size() < 2)
      break;

    std::unordered_map<TokenPair, uint32_t> TokenCount;
    uint32_t                                MaxCount = 0;
    TokenPair                               BestPair{0, 0};

    for (size_t i = 0; i < tokens.size() - 1; i++) {
      if (tokens[i] == LIMITERCODE || tokens[i + 1] == LIMITERCODE)
        continue;

      TokenPair currentPair = {tokens[i], tokens[i + 1]};
      uint32_t  count       = ++TokenCount[currentPair];

      if (count > MaxCount) {
        MaxCount = count;
        BestPair = currentPair;
      }
    }

    if (MaxCount == 0)
      break;

    uint32_t newTokenId = 256 + pass;
    vocab[newTokenId]   = vocab[BestPair.First] + vocab[BestPair.Second];

    std::vector<uint32_t> newTokens;
    newTokens.reserve(tokens.size());

    for (size_t i = 0; i < tokens.size();) {
      if (i + 1 < tokens.size() && tokens[i] == BestPair.First &&
          tokens[i + 1] == BestPair.Second) {
        newTokens.push_back(newTokenId);
        i += 2;
      } else {
        newTokens.push_back(tokens[i]);
        i++;
      }
    }
    tokens = std::move(newTokens);
  }
}
}; // namespace Odin
