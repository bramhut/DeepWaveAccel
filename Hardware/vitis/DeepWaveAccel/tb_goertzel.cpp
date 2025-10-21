#include "tb_base.hpp"
#include "goertzel.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <complex>

int tb_goertzel() {
    hls::stream<AxisWordSampleIn> in_stream;
    hls::stream<AxisWordDFTc> out_stream;
    goertzel_config cfg;

    // Read WAV file
    const char* filename = "C:/Users/bramh/Documents/Github/DeepWaveAccel/Simulation/FRIDA/FRIDA/recordings/20160908/data_pyramic/segmented/two_speakers/1-5.wav";
    std::vector<int16_t> wav_samples;
    int channels, samplerate, frames;
    if(!read_wav_16bit(filename, wav_samples, channels, samplerate, frames)) {
        std::cerr << "Failed to read WAV file or unsupported format" << std::endl;
        return 1;
    }

    if (channels != N_ELEM){
        std::cerr << "Number of channels in input wave file does not match DeepWaveAccel setup" << std::endl;
        return 1;
    }

    // Prepare configuration
    std::cout << "sample rate: " << samplerate <<std::endl;
    goertzel_prepare_config(cfg, (double)samplerate, FF);

    std::cout << "Calculated Goertzel coefficients:" << std::endl;
    for (int b = 0; b < NBINS; b++) {
        std::cout << "  cos_omega["  << b << "] = " << cfg.COS_OMEGA[b].to_double()
                  << ", cos_omega2[" << b << "] = " << cfg.COS_OMEGA2[b].to_double()
                  << ", sin_omega["  << b << "] = " << cfg.SIN_OMEGA[b].to_double() <<std::endl;
    }

    int n_batches = frames / N_WIN;

    // Stream input sample by sample, element by element
    for(int b = 0; b < n_batches; ++b) {
        for(int ch = 0; ch < N_ELEM; ++ch) {
            for(int n = 0; n < N_WIN; ++n) {
                int idx = (b * N_WIN + n) * N_ELEM + ch; // interleaved
                AxisWordSampleIn t;
                t.data = 16.0 * double(wav_samples[idx]) / 32768.0; // apply gain
                t.last = false;
                in_stream.write(t);
            }
        }
    }

    // Run Goertzel kernel
    int total_samples = n_batches * N_WIN * N_ELEM;
    for(int i = 0; i < total_samples; ++i) {
        goertzel(in_stream, out_stream, cfg);
    }

    // Collect outputs into array [N_ELEM][n_batches]
    std::vector<std::vector<DFTc_t>> goertzel_out(N_ELEM, std::vector<DFTc_t>(n_batches));

    for(int b = 0; b < n_batches; ++b) {
        for(int ch = 0; ch < N_ELEM; ++ch) {
            if(!out_stream.empty()) {
                AxisWordDFTc in = out_stream.read();
                goertzel_out[ch][b] = std::complex<DFT_t>(in.re, in.im);
            } else {
                std::cerr << "Trying to read non-existing AXIS output value, THIS SHOULDN'T HAPPEN" << std::endl;
            }
        }
    }

    // Write results to CSV file
    std::string file_sim_out =  "/goertzel_sim.csv";
    std::ofstream csv(std::string(OUTPUT_DIR) + file_sim_out);
    if (!csv.is_open()) {
        std::cerr << "Failed to open output CSV file" << std::endl;
        return 1;
    }

    // Optional header
    csv << "batch,channel,real,imag\n";

    for (int b = 0; b < n_batches; ++b) {
        for (int ch = 0; ch < N_ELEM; ++ch) {
            auto& out = goertzel_out[ch][b];
            csv << b << "," << ch << ","
                << out.real().to_double() << ","
                << out.imag().to_double() << "\n";
        }
    }


    csv.close();
    std::cout << "Wrote " << N_ELEM * n_batches << " results to \"output" << file_sim_out << "\"" << std::endl;

    std::cout << "Done." << std::endl;
    return 0;
}
