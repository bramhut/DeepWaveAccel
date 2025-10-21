#pragma once
#include "types.hpp"
#include "pair_rom_data.hpp"
#include <vector>
#include <cstdint>

bool read_wav_16bit(const char* filename, std::vector<int16_t>& samples, int& channels, int& samplerate, int& frames);