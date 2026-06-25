#pragma once
#include <algorithm>
#include <cstdint>
#include <iostream>
#include "logging.hpp"
#include <sys/ioctl.h>
#include <unistd.h>

// Just for logo . Contains many hardcoded values .
bool sample(const std::string& logo , float u, float v) {
  int x = std::clamp(int(u * 29), 0, 28);
  int y = std::clamp(int(v * 5), 0, 4);
  return logo[y * 29 + x] == '1';
}

uint32_t clamp255(float x) {
  return std::clamp(uint32_t(x * 255.0), 0u, 255u);
}

void printLogo() {
  const std::string logo =
    "01111110011111100110111000011"
    "11000011011000110110111100011"
    "11000011011000110110110110011"
    "11000011011000110110110011011"
    "01111110011111100110110001111";

  winsize ws{};
  ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws);

  uint32_t target_w = ws.ws_col / 1.5;
  uint32_t target_h = target_w * 5 / 29;

  uint32_t start_col = (ws.ws_col - target_w) / 2 + 1;

  for (uint32_t y = 0; y < target_h; ++y) {
    std::cout << "\x1b[" << (y + 1) << ';' << start_col << 'H';
    for (uint32_t x = 0; x < target_w; ++x) {

      double u = double(x) / (target_w - 1);
      double v = double(y) / (target_h - 1);

      if (sample(logo,u, v)) {

        uint32_t r = clamp255(0.05 * (1.0 - v));
        uint32_t g = clamp255(v);              
        uint32_t b = clamp255(1.0);             

        std::cout << "\x1b[38;2;"
          << r << ";"
          << g << ";"
          << b << "m"
          << "█";
      } else {
        std::cout << " ";
      }
    }
    std::cout << "\x1b[0m\n";
  }
}

void PrintHome(){
  printLogo();
  static const auto usage =
    "Usage:\n"
    "\t./odin --model <model_path> --tokeniser-json <tokeniser_json_path>\n"
    "\t       [--thread <num_threads>]\n"
    "\t       [--ipc-path <path>] [--use-ipc <0|1>]\n"
    "\t       [--history <history_file>]\n\n"
    "Options:\n"
    "\t--model            Path to the model file (required)\n"
    "\t--tokeniser-json   Path to the tokenizer JSON file (required)\n"
    "\t--thread           Number of threads to use (default: system dependent)\n"
    "\t--ipc-path         Path or endpoint for IPC input/output\n"
    "\t--use-ipc          Enable IPC mode (0 = disabled, 1 = enabled)\n"
    "\t--history          Path to save/load conversation history";

  Log(usage);
}
