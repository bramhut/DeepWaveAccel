#pragma once
#include "types.hpp"
#include "pair_rom_data.hpp"
#include <vector>
#include <cstdint>
#include <string>

bool read_wav_16bit(std::string filename, std::vector<int16_t>& samples, int& channels, int& samplerate, int& frames);