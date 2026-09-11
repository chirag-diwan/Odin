#include "config.hpp"
#include "app.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    return -1;
  }

  Config config = ParseConfig(argc, argv);

  odin::App app;

  app.InitBase(config);
  app.ConfigureChat();
  app.Run();
  app.Delete();

  return 0;
}
