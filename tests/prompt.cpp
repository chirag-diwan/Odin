#include <iostream>
#include <string>
#include "../external/replxx/include/replxx.hxx"

std::string run_inference_sync(const std::string& input) {
  return "Model output for: " + input;
}

int main() {
  replxx::Replxx rx;

  rx.install_window_change_handler();
  rx.set_max_history_size(1000);

  while (true) {
    const char* c_input = rx.input(" $ ");

    if (c_input == nullptr) {
      std::cout << "\nBye!\n";
      break;
    }

    std::string user_input{c_input};

    // Ignore empty submissions
    if (user_input.empty()) {
      continue;
    }

    rx.history_add(user_input);

    // Exit condition
    if (user_input == "exit" || user_input == "quit") {
      break;
    }

    // 4. Blocking Execution
    // The terminal is currently parked at the beginning of a new line.
    std::string response = run_inference_sync(user_input);

    // Print the result. Standard output is safe to use here because 
    // replxx is completely dormant.
    std::cout << response << "\n";
  }

  return 0;
}
