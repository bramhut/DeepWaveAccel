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
// int tb_goertzel();
// int tb_crosscor();
// int tb_backproj();
// int tb_deblur();

int main() {
    std::cout.setf(std::ios::unitbuf);
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
    // Settings

    int max_frames = -1;  // Set to -1 to process all frames (takes around 8min currently)

    // -------------------------------------------------------------------------
    // Define AXIS streams (new interfaces)
    // -------------------------------------------------------------------------
    hls::stream<sample_t>  in_stream;
    hls::stream<out_axis_t> out_stream;

    goertzel_config goer_cfg;
    deblur_config   debl_cfg;

    std::cout << "----------------------------------------------\n";
    std::cout << "     DeepWaveAccel full pipeline test\n";
    std::cout << "----------------------------------------------\n";

    // -------------------------------------------------------------------------
    // Load steering vectors (DDR-style array: IMG_LEN x N_ELEM, pixel-major)
    // -------------------------------------------------------------------------
    std::string b_file = std::string(PARAM_DIR) + "/b_vectors.csv";
    std::ifstream b_in(b_file);
    if (!b_in.is_open()) {
        std::cerr << "Failed to open steering vector file: " << b_file << std::endl;
        return 1;
    }

    std::string header;
    std::getline(b_in, header); // skip header

    // Flat DDR buffer in pixel-major order: idx = pixel*N_ELEM + elem
    std::vector<b_t> b_ddr(IMG_LEN * N_ELEM);
    int pixel, elem;
    double bre, bim;
    int b_count = 0;
    while (b_in >> pixel) {
        char comma;
        b_in >> comma >> elem >> comma >> bre >> comma >> bim;
        if (pixel < 0 || pixel >= IMG_LEN || elem < 0 || elem >= N_ELEM) {
            std::cerr << "b_vectors.csv index out of bounds: pixel=" << pixel
                      << ", elem=" << elem << std::endl;
            return 1;
        }
        b_ddr[pixel * N_ELEM + elem] = b_t((b_real_t)bre, (b_real_t)bim);
        ++b_count;
    }
    b_in.close();
    std::cout << "[DeepWave] Loaded " << b_count << " steering vector entries ("
              << "expected " << (IMG_LEN * N_ELEM) << ")\n";

    // -------------------------------------------------------------------------
    // Load tau compensation (DDR-style array: IMG_LEN entries)
    // -------------------------------------------------------------------------
    std::string tau_file = std::string(PARAM_DIR) + "/tau.csv";
    std::ifstream tau_in(tau_file);
    if (!tau_in.is_open()) {
        std::cerr << "Failed to open tau file: " << tau_file << std::endl;
        return 1;
    }
    std::getline(tau_in, header);

    std::vector<tau_t> tau_ddr(IMG_LEN);
    double tau_val;
    int tau_count = 0;
    while (tau_in >> tau_val) {
        if (tau_count >= IMG_LEN) {
            std::cerr << "tau.csv has more than IMG_LEN entries\n";
            return 1;
        }
        tau_ddr[tau_count++] = (tau_t)tau_val;
    }
    tau_in.close();
    std::cout << "[DeepWave] Loaded " << tau_count << " tau compensation values ("
              << "expected " << IMG_LEN << ")\n";

    // -------------------------------------------------------------------------
    // Load Laplacian data (DDR-style array: main + ND*IMG_LEN)
    // Layout must be: [main_diag] then [ND rows each of IMG_LEN]
    // -------------------------------------------------------------------------
    std::string lap_file = std::string(PARAM_DIR) + "/laplacian.csv";
    std::ifstream lap_in(lap_file);
    if (!lap_in.is_open()) {
        std::cerr << "Failed to open Laplacian file: " << lap_file << std::endl;
        return 1;
    }
    std::getline(lap_in, header);

    std::vector<lap_t> lap_ddr;
    lap_ddr.reserve(1 + ND * IMG_LEN);

    double lap_val;
    int lap_count = 0;
    while (lap_in >> lap_val) {
        lap_ddr.push_back((lap_t)lap_val);
        ++lap_count;
    }
    lap_in.close();

    if (lap_count != 1 + ND * IMG_LEN) {
        std::cerr << "[DeepWave] Laplacian length mismatch. Got " << lap_count
                  << ", expected " << (1 + ND * IMG_LEN) << " (main + ND*IMG_LEN)\n";
        return 1;
    }
    std::cout << "[DeepWave] Loaded Laplacian: main + " << ND << " off-diagonals\n";

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
    std::cout << "[DeepWave] Loaded " << ND << " Laplacian offsets\n";

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
    std::cout << "[DeepWave] Loaded theta coefficients (K=" << (int)debl_cfg.K << ")\n";

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
                  << ") does not match setup (" << N_ELEM << ")\n";
        return 1;
    }

    goertzel_prepare_config(goer_cfg, (double)samplerate, FF);

    int n_batches = n_sample / N_WIN;
    int n_batches_group_aligned = (n_batches / GROUP_FRAMES) * GROUP_FRAMES;
    int expected_frames = n_batches_group_aligned / GROUP_FRAMES;

    std::cout << "[DeepWave] Input WAV: " << n_sample << " samples @ "
              << samplerate << " Hz, " << channels << " channels, "
              << n_batches << " Goertzel batches, of which "
              << n_batches_group_aligned << " will be processed "
              << "(aligned to GROUP_FRAMES)\n";

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
    std::cout << "[DeepWave] Streamed all input samples\n";

    // -------------------------------------------------------------------------
    // Run full kernel (DATAFLOW pipeline)
    //   Each frame produces (IMG_LEN + 1) 32-bit words: [norm, pixel0..pixelN]
    // -------------------------------------------------------------------------
    std::cout << "[DeepWave] Starting pipeline execution...\n";
    auto start = high_resolution_clock::now();

    int last_frame_count = 0;
    auto last_time = start;

    while (true) {
        deepwaveaccel(in_stream,
                    b_ddr.data(),
                    tau_ddr.data(),
                    lap_ddr.data(),
                    out_stream,
                    goer_cfg,
                    debl_cfg);

        int frames_done = (int)(out_stream.size() / (IMG_LEN + 1));

        if (frames_done > last_frame_count) {
            auto now = high_resolution_clock::now();
            auto delta = duration_cast<milliseconds>(now - last_time).count() / 1000.0;
            auto elapsed = duration_cast<milliseconds>(now - start).count() / 1000.0;

            std::cout << "  [Frame " << frames_done
                    << "/" << expected_frames
                    << "]  dt = " << std::fixed << std::setprecision(2)
                    << delta << " s  (total = "
                    << std::setprecision(2) << elapsed << " s)\n";

            last_time = now;
            last_frame_count = frames_done;
        }

        if (frames_done >= ((max_frames < 0) ? expected_frames : std::min(expected_frames, max_frames))) break;
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << "[DeepWave] Finished in "
            << std::fixed << std::setprecision(2)
            << duration.count() / 1000.0 << " s\n";


    // -------------------------------------------------------------------------
    // Collect outputs: parse frames (norm + pixels)
    // -------------------------------------------------------------------------
    std::vector<norm_sum_t> norms;
    norms.reserve(expected_frames);

    std::vector<img_t> image_out;
    image_out.reserve((size_t)expected_frames * IMG_LEN);

    while (!out_stream.empty()) {
        out_axis_t w_norm = out_stream.read();
        norm_sum_t nv;
        nv.range() = w_norm.range(norm_sum_t::width-1, 0); 
        norms.push_back(nv);

        for (int i = 0; i < IMG_LEN; ++i) {
            img_t pix;
            out_axis_t w_pix = out_stream.read();
            pix.range() = w_pix.range(img_t::width-1, 0);
            image_out.push_back(pix);
        }
    }

    std::cout << "[DeepWave] Collected " << image_out.size()
              << " pixels (" << (image_out.size()/IMG_LEN) << " frames)\n";
    std::cout << "[DeepWave] Collected " << norms.size() << " norm values\n";

    // -------------------------------------------------------------------------
    // Save final output (pixels only; norms optional)
    // -------------------------------------------------------------------------
    std::string file_out = std::string(OUTPUT_DIR) + "/deepwave_sim.csv";
    std::ofstream csv_out(file_out);
    if (!csv_out.is_open()) {
        std::cerr << "Failed to open output file: " << file_out << std::endl;
        return 1;
    }

    csv_out << "frame,pixel,value\n";
    for (size_t i = 0; i < image_out.size(); ++i) {
        int frame = (int)(i / IMG_LEN);
        int pix   = (int)(i % IMG_LEN);
        double scaled = image_out[i].to_double() / (2048.0 * std::tanh(1.0));
        csv_out << frame << "," << pix << "," << scaled << "\n";
    }
    csv_out.close();
    std::cout << "[DeepWave] Wrote final pixel output (scaled) to " << file_out << "\n";

    // Optional: also write norms
    std::string norms_out = std::string(OUTPUT_DIR) + "/deepwave_norms.csv";
    std::ofstream csv_norm(norms_out);
    if (csv_norm.is_open()) {
        csv_norm << "frame,norm\n";
        for (size_t f = 0; f < norms.size(); ++f) {
            csv_norm << f << "," << norms[f].to_double() << "\n";
        }
        csv_norm.close();
        std::cout << "[DeepWave] Wrote norms to " << norms_out << "\n";
    }

    std::cout << "Done.\n";
    return 0;
}
