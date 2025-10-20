#include "tb_base.hpp"
#include <iostream>
#include <fstream>

// Minimal WAV reader for 16-bit PCM, 48 channels
bool read_wav_16bit(const char* filename, std::vector<int16_t>& samples, int& channels, int& samplerate, int& frames) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) return false;

    char riff[4];
    f.read(riff, 4);
    if (std::string(riff,4) != "RIFF") return false;
    f.ignore(4); // file size
    char wave[4];
    f.read(wave,4);
    if (std::string(wave,4) != "WAVE") return false;

    // Read chunks
    while(f) {
        char chunk_id[4];
        f.read(chunk_id,4);
        uint32_t chunk_size;
        f.read(reinterpret_cast<char*>(&chunk_size), 4);

        if(std::string(chunk_id,4) == "fmt ") {
            uint16_t audio_format, num_channels, bits_per_sample;
            uint32_t sample_rate, byte_rate;
            uint16_t block_align;
            f.read(reinterpret_cast<char*>(&audio_format),2);
            f.read(reinterpret_cast<char*>(&num_channels),2);
            f.read(reinterpret_cast<char*>(&sample_rate),4);
            f.read(reinterpret_cast<char*>(&byte_rate),4);
            f.read(reinterpret_cast<char*>(&block_align),2);
            f.read(reinterpret_cast<char*>(&bits_per_sample),2);
            if(audio_format != 1 || bits_per_sample != 16) return false;
            channels = num_channels;
            samplerate = sample_rate;
            f.ignore(chunk_size - 16);
        } else if(std::string(chunk_id,4) == "data") {
            frames = chunk_size / (channels * 2);
            samples.resize(frames * channels);
            f.read(reinterpret_cast<char*>(samples.data()), chunk_size);
            return true;
        } else {
            f.ignore(chunk_size);
        }
    }
    return false;
}