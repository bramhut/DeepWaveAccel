#include "tb_base.hpp"
#include "top.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>
#include <complex>
#include <chrono>

using namespace std::chrono;


int tb_deepwaveaccel();
int tb_goertzel();
int tb_crosscor();
int tb_backproj();
int tb_deblur();

int main() {
    std::cout.setf(std::ios::unitbuf); // Weird cout lines dissappearing... Not sure if this helpes
    std::setvbuf(stdout, NULL, _IONBF, 0);

    tb_deepwaveaccel();
    // tb_goertzel();
    // tb_crosscor();
    // tb_backproj();
    // tb_deblur();

}


// -----------------------------------------------------------------------------
// DeepWaveAccel – Full-pipeline testbench
// -----------------------------------------------------------------------------
int tb_deepwaveaccel() {
    // -------------------------------------------------------------------------
    // Define all AXIS streams
    // -------------------------------------------------------------------------
    hls::stream<sample_t> in_stream;
    hls::stream<b_t>              b_stream;
    hls::stream<tau_t>            tau_stream;
    hls::stream<lap_t>            lap_stream;
    hls::stream<img_t>      out_stream;
    hls::stream<norm_sum_t>       norm_stream;

    goertzel_config goer_cfg;
    deblur_config   debl_cfg;

    std::cout << "----------------------------------------------\n";
    std::cout << "     🌀 DeepWaveAccel full pipeline test\n";
    std::cout << "----------------------------------------------\n";

    // -------------------------------------------------------------------------
    // Load steering vectors (b_stream)
    // -------------------------------------------------------------------------
    std::string b_file = std::string(PARAM_DIR) + "/b_vectors.csv";
    std::ifstream b_in(b_file);
    if (!b_in.is_open()) {
        std::cerr << "Failed to open steering vector file: " << b_file << std::endl;
        return 1;
    }

    std::string header;
    std::getline(b_in, header); // skip header

    int pixel, elem;
    double bre, bim;
    int b_count = 0;
    while (b_in >> pixel) {
        char comma;
        b_in >> comma >> elem >> comma >> bre >> comma >> bim;
        b_stream.write(b_t((b_real_t)bre, (b_real_t)bim));
        ++b_count;
    }
    b_in.close();
    std::cout << "[DeepWave] Loaded " << b_count << " steering vector entries" << std::endl;

    // -------------------------------------------------------------------------
    // Load tau compensation (tau_stream)
    // -------------------------------------------------------------------------
    std::string tau_file = std::string(PARAM_DIR) + "/tau.csv";
    std::ifstream tau_in(tau_file);
    if (!tau_in.is_open()) {
        std::cerr << "Failed to open tau file: " << tau_file << std::endl;
        return 1;
    }
    std::getline(tau_in, header);
    double tau_val;
    int tau_count = 0;
    while (tau_in >> tau_val) {
        tau_stream.write((tau_t)tau_val);
        ++tau_count;
    }
    tau_in.close();
    std::cout << "[DeepWave] Loaded " << tau_count << " tau compensation values" << std::endl;

    // -------------------------------------------------------------------------
    // Load Laplacian data (lap_stream)
    // -------------------------------------------------------------------------
    std::string lap_file = std::string(PARAM_DIR) + "/laplacian.csv";
    std::ifstream lap_in(lap_file);
    if (!lap_in.is_open()) {
        std::cerr << "Failed to open Laplacian file: " << lap_file << std::endl;
        return 1;
    }
    std::getline(lap_in, header);
    double lap_val;
    int d = 0, i = 0;
    while (lap_in >> lap_val) {
        lap_stream.write((lap_t)lap_val);
        ++i;
        if (i == IMG_LEN) {
            i = 0;
            ++d;
        }
    }
    lap_in.close();

    std::cout << "[DeepWave] Loaded Laplacian: main + " << d << " off-diagonals" << std::endl;

    // -------------------------------------------------------------------------
    // Load Laplacian offsets (AXI-Lite config)
    // -------------------------------------------------------------------------
    debl_cfg.n_layers = 5;
    debl_cfg.K = 22;

    std::string offset_file = std::string(PARAM_DIR) + "/lap_offsets.csv";
    std::ifstream off_in(offset_file);
    if (!off_in.is_open()) {
        std::cerr << "Failed to open Laplacian offset file: " << offset_file << std::endl;
        return 1;
    }

    std::string token;
    for (int k = 0; k < ND; ++k) {
        if (!std::getline(off_in, token, (k == ND - 1 ? '\n' : ','))) {
            std::cerr << "Not enough offsets in file!" << std::endl;
            return 1;
        }
        try {
            debl_cfg.lap_off[k] = (idx_t)std::stod(token);
        } catch (...) {
            std::cerr << "Invalid offset format in file!" << std::endl;
            return 1;
        }
    }
    off_in.close();
    std::cout << "[DeepWave] Loaded " << ND << " Laplacian offsets" << std::endl;

    // -------------------------------------------------------------------------
    // Load θ coefficients (AXI-Lite config)
    // -------------------------------------------------------------------------
    std::string theta_file = std::string(PARAM_DIR) + "/theta.csv";
    std::ifstream th_in(theta_file);
    if (!th_in.is_open()) {
        std::cerr << "Failed to open theta file: " << theta_file << std::endl;
        return 1;
    }

    for (int k = 0; k <= MAX_ORDER; ++k) {
        if (!std::getline(th_in, token, (k == MAX_ORDER ? '\n' : ','))) {
            std::cerr << "Not enough theta coefficients loaded!" << std::endl;
            return 1;
        }
        try {
            debl_cfg.theta[k] = (theta_t)std::stod(token);
        } catch (...) {
            std::cerr << "Invalid theta coefficient format in file!" << std::endl;
            return 1;
        }
    }
    th_in.close();
    std::cout << "[DeepWave] Loaded theta coefficients (K=" << (int)debl_cfg.K << ")" << std::endl;


    // -------------------------------------------------------------------------
    // Load Goertzel input (from WAV)
    // -------------------------------------------------------------------------
    std::string wave_file = std::string(WAVE_DIR) + "/two_speakers/1-5.wav";
    std::vector<int16_t> wav_samples;
    int channels, samplerate, n_sample;
    if(!read_wav_16bit(wave_file, wav_samples, channels, samplerate, n_sample)) {
        std::cerr << "Failed to read WAV file or unsupported format" << std::endl;
        return 1;
    }

    if (channels != N_ELEM) {
        std::cerr << "Number of channels in input wave file (" << channels
                  << ") does not match setup (" << N_ELEM << ")" << std::endl;
        return 1;
    }

    goertzel_prepare_config(goer_cfg, (double)samplerate, FF);

    int n_batches = n_sample / N_WIN;
    int n_batches_group_aligned = (n_batches / GROUP_FRAMES) * GROUP_FRAMES;
    std::cout << "[DeepWave] Input WAV: " << n_sample << " samples @ "
              << samplerate << " Hz, " << channels << " channels, "
              << n_batches << " Goertzel batches, of which " << n_batches_group_aligned << " will be processed (aligned to GROUP_FRAMES)\n";

    // Apply gain and stream to Goertzel input
    for (int b = 0; b < n_batches_group_aligned; ++b) {
        for (int ch = 0; ch < N_ELEM; ++ch) {
            for (int n = 0; n < N_WIN; ++n) {
                int idx = (b * N_WIN + n) * N_ELEM + ch;
                sample_t t = 16.0 * double(wav_samples[idx]) / 32768.0;
                in_stream.write(t);
            }
        }
    }
    std::cout << "[DeepWave] Streamed all input samples" << std::endl;

    // -------------------------------------------------------------------------
    // Run full kernel (DATAFLOW pipeline)
    // -------------------------------------------------------------------------
    std::cout << "[DeepWave] Starting pipeline execution..." << std::endl;
    auto start = high_resolution_clock::now();

    // Drive one tick per iteration until final image appears
    int frame_idx = 0;
    while (out_stream.size() / IMG_LEN < n_batches / GROUP_FRAMES) {
        deepwaveaccel(in_stream, b_stream, tau_stream, lap_stream,
                      out_stream, norm_stream, goer_cfg, debl_cfg);
        frame_idx = out_stream.size() / IMG_LEN;
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << "[DeepWave] Finished in " << duration.count()/1000.0 << " s" << std::endl;

    // -------------------------------------------------------------------------
    // Collect outputs
    // -------------------------------------------------------------------------
    std::vector<img_t> image_out;
    while (!out_stream.empty()) {
        image_out.push_back(out_stream.read());
    }

    std::cout << "[DeepWave] Collected " << image_out.size()
              << " pixels (" << image_out.size()/IMG_LEN << " frames)" << std::endl;

    // -------------------------------------------------------------------------
    // Save final output
    // -------------------------------------------------------------------------
    std::string file_out = std::string(OUTPUT_DIR) + "/deepwave_sim.csv";
    std::ofstream csv_out(file_out);
    if (!csv_out.is_open()) {
        std::cerr << "Failed to open output file: " << file_out << std::endl;
        return 1;
    }

    csv_out << "frame,pixel,value\n";
    for (size_t i = 0; i < image_out.size(); ++i) {
        int frame = i / IMG_LEN;
        int pix = i % IMG_LEN;
        double scaled = image_out[i].to_double() / (2048.0 * std::tanh(1.0));
        csv_out << frame << "," << pix << "," << scaled << "\n";
    }
    csv_out.close();

    std::cout << "[DeepWave] Wrote final output (scaled) to " << file_out << std::endl;
    std::cout << "✅ Done.\n";
    return 0;
}
