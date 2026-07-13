#pragma once
#include <cstdint>
#include <string>
#include <sys/ioctl.h>
#include <unistd.h>

// Just for logo . Contains many hardcoded values .
bool sample(const std::string& logo , float u, float v) ;
uint32_t clamp255(float x) ;
void printLogo() ;
void PrintHome();
