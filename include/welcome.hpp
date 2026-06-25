#pragma once
#include <algorithm>
#include <iostream>
#include <sys/ioctl.h>
#include <unistd.h>

const std::string logo_stencil =
"01111110011111100110111000011"
"11000011011000110110111100011"
"11000011011000110110110110011"
"11000011011000110110110011011"
"01111110011111100110110001111";

bool sample(double u, double v) {
  int x = std::clamp(int(u * 29), 0, 28);
  int y = std::clamp(int(v * 5), 0, 4);
  return logo_stencil[y * 29 + x] == '1';
}

int clamp255(double x) {
  return std::clamp(int(x * 255.0), 0, 255);
}

void printLogo() {
  winsize ws{};
  ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws);

  int target_h = ws.ws_row / 2.5;
  int target_w = ws.ws_col / 1.5;

  for (int y = 0; y < target_h; ++y) {
    for (int x = 0; x < target_w; ++x) {

      double u = double(x) / (target_w - 1);
      double v = double(y) / (target_h - 1);

      if (sample(u, v)) {

        int r = clamp255(0.05 * (1.0 - v));   
        int g = clamp255(v);                  
        int b = clamp255(1.0);                
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



const auto usage =
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
"\t--history          Path to save/load conversation history\n";
