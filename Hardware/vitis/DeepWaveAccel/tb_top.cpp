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
// DeepWaveAccel – Full-pipeline testbench (param-stream version)
// -----------------------------------------------------------------------------
int tb_deepwaveaccel() {
    // Settings
    int max_frames = -1;  // -1 → process all

    // -------------------------------------------------------------------------
    // AXIS streams
    // -------------------------------------------------------------------------
    hls::stream<word_t>     in_stream;
    hls::stream<word_t> param_bp_stream;
    hls::stream<word_t> param_db_stream;
    hls::stream<out_axis_t>   out_stream;

    goertzel_config goer_cfg;
    deblur_config   debl_cfg;
    debl_cfg.n_layers = 5;

    status_gz_t status_gz;
    status_cc_t status_cc;
    status_bp_t status_bp;
    status_db_t status_db;

    std::cout << "----------------------------------------------\n";
    std::cout << "     DeepWaveAccel full pipeline test\n";
    std::cout << "----------------------------------------------\n";

    // -------------------------------------------------------------------------
    // 1. Load parameters from CSVs
    // -------------------------------------------------------------------------
    std::string header;

    // --- b_vectors.csv ---
    std::string b_file = std::string(PARAM_DIR) + "/b_vectors.csv";
    std::ifstream b_in(b_file);
    if (!b_in.is_open()) {
        std::cerr << "Failed to open steering vector file: " << b_file << std::endl;
        return 1;
    }
    std::getline(b_in, header);
    std::vector<b_t> b_ddr(IMG_LEN * N_ELEM);
    {
        int pixel, elem;
        double bre, bim;
        int b_count = 0;
        while (b_in >> pixel) {
            char comma;
            b_in >> comma >> elem >> comma >> bre >> comma >> bim;
            b_ddr[pixel * N_ELEM + elem] = b_t((b_real_t)bre, (b_real_t)bim);
            ++b_count;
        }
        std::cout << "[DeepWave] Loaded " << b_count << " steering vector entries\n";
    }
    b_in.close();

    // --- tau.csv ---
    std::string tau_file = std::string(PARAM_DIR) + "/tau.csv";
    std::ifstream tau_in(tau_file);
    if (!tau_in.is_open()) {
        std::cerr << "Failed to open tau file: " << tau_file << std::endl;
        return 1;
    }
    std::getline(tau_in, header);
    std::vector<tau_t> tau_ddr(IMG_LEN);
    {
        double tau_val;
        int i = 0;
        while (tau_in >> tau_val && i < IMG_LEN)
            tau_ddr[i++] = (tau_t)tau_val;
    }
    tau_in.close();
    std::cout << "[DeepWave] Loaded " << tau_ddr.size() << " tau values\n";

    // --- laplacian.csv ---
    std::string lap_file = std::string(PARAM_DIR) + "/laplacian.csv";
    std::ifstream lap_in(lap_file);
    if (!lap_in.is_open()) {
        std::cerr << "Failed to open Laplacian file: " << lap_file << std::endl;
        return 1;
    }
    std::getline(lap_in, header);
    std::vector<lap_t> lap_ddr;
    {
        double v;
        while (lap_in >> v)
            lap_ddr.push_back((lap_t)v);
    }
    lap_in.close();
    std::cout << "[DeepWave] Loaded Laplacian: " << lap_ddr.size() << " entries\n";

    // --- lap_offsets.csv ---
    std::string offset_file = std::string(PARAM_DIR) + "/lap_offsets.csv";
    std::ifstream off_in(offset_file);
    if (!off_in.is_open()) {
        std::cerr << "Failed to open Laplacian offset file: " << offset_file << std::endl;
        return 1;
    }

    std::vector<idx_t> lap_offs;
    std::string line;
    if (std::getline(off_in, line)) {
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            try {
                int v = std::stoi(token);
                lap_offs.push_back((idx_t)v);
            } catch (...) {
                std::cerr << "Invalid Laplacian offset: " << token << std::endl;
                return 1;
            }
        }
    }
    off_in.close();

    if ((int)lap_offs.size() != ND) {
        std::cerr << "[DeepWave] Warning: Expected " << ND
                << " Laplacian offsets but got " << lap_offs.size() << "\n";
    }

    std::cout << "[DeepWave] Loaded Laplacian offsets ("
            << lap_offs.size() << " values)\n";


    // --- theta.csv ---
    std::string theta_file = std::string(PARAM_DIR) + "/theta.csv";
    std::ifstream th_in(theta_file);
    if (!th_in.is_open()) {
        std::cerr << "Failed to open theta file: " << theta_file << std::endl;
        return 1;
    }

    std::vector<theta_t> theta_vals;
    if (std::getline(th_in, line)) {
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            try {
                double v = std::stod(token);
                theta_vals.push_back((theta_t)v);
            } catch (...) {
                std::cerr << "Invalid theta coefficient: " << token << std::endl;
                return 1;
            }
        }
    }
    th_in.close();

    int K = (int)theta_vals.size() - 1;
    std::cout << "[DeepWave] Loaded " << theta_vals.size()
            << " theta coefficients (K=" << K << ")\n";


    // -------------------------------------------------------------------------
    // 2. Write parameters into param_stream
    // -------------------------------------------------------------------------
    {
        word_t p;

        // --- Backprojection (user=1) ---
        for (int pix = 0; pix < IMG_LEN; ++pix) {
            for (int e = 0; e < N_ELEM; ++e) {
                const b_t &b = b_ddr[pix * N_ELEM + e];

                // real
                p.range() = b.real().range();
                param_bp_stream.write(p);

                // imag
                p.range() = b.imag().range();
                param_bp_stream.write(p);
            }
        }

        for (int pix = 0; pix < IMG_LEN; ++pix) {
            p.range() = tau_ddr[pix].range();
            param_bp_stream.write(p);
        }

        // --- Deblur (user=0) ---
        // K
        p = K;
        param_db_stream.write(p);

        // theta
        for (int i = 0; i <= K; ++i) {
            p.range() = theta_vals[i].range();
            param_db_stream.write(p);
        }

        // offsets
        for (int i = 0; i < ND; ++i) {
            p = (ap_uint<32>)lap_offs[i];
            param_db_stream.write(p);
        }

        // lap_main
        p.range() = lap_ddr[0].range();
        param_db_stream.write(p);

        // lap_rest (ND × IMG_LEN)
        for (int d = 0; d < ND; ++d)
            for (int i = 0; i < IMG_LEN; ++i) {
                p.range() = lap_ddr[1 + d * IMG_LEN + i].range();
                param_db_stream.write(p);
            }
    }

    // -------------------------------------------------------------------------
    // 3. Load WAV data (same as before)
    // -------------------------------------------------------------------------
    std::string wave_file = std::string(WAVE_DIR) + "/two_speakers/1-5.wav";
    std::vector<int16_t> wav_samples;
    int channels, samplerate, n_sample;
    if (!read_wav_16bit(wave_file, wav_samples, channels, samplerate, n_sample)) {
        std::cerr << "Failed to read WAV\n";
        return 1;
    }

    if (channels != N_ELEM) {
        std::cerr << "Invalid channel count\n";
        return 1;
    }

    goertzel_prepare_config(goer_cfg, (double)samplerate, FF);
    std::cout << "[Deepwave] Goertzel config bin0: COS_OMEGA[0]: " << goer_cfg.COS_OMEGA[0]  << " (" << goer_cfg.COS_OMEGA[0].range()  << "), COS_OMEGA2[0]: " << goer_cfg.COS_OMEGA2[0] << " (" << goer_cfg.COS_OMEGA2[0].range() << "), SIN_OMEGA[0]: " << goer_cfg.SIN_OMEGA[0] << " (" << goer_cfg.SIN_OMEGA[0].range() << ")" << std::endl;
    std::cout << "[Deepwave] Goertzel config bin1: COS_OMEGA[1]: " << goer_cfg.COS_OMEGA[1]  << " (" << goer_cfg.COS_OMEGA[1].range()  << "), COS_OMEGA2[1]: " << goer_cfg.COS_OMEGA2[1] << " (" << goer_cfg.COS_OMEGA2[1].range() << "), SIN_OMEGA[1]: " << goer_cfg.SIN_OMEGA[1] << " (" << goer_cfg.SIN_OMEGA[1].range() << ")" << std::endl;
    int n_batches = n_sample / N_WIN;
    int n_batches_group_aligned = (n_batches / GROUP_FRAMES) * GROUP_FRAMES;
    int expected_frames = n_batches_group_aligned / GROUP_FRAMES;

    for (int b = 0; b < n_batches_group_aligned; ++b)
        for (int ch = 0; ch < N_ELEM; ++ch)
            for (int n = 0; n < N_WIN; ++n) {
                int idx = (b * N_WIN + n) * N_ELEM + ch;
                sample_t t = 16.0 * double(wav_samples[idx]) / 32768.0;
                word_t out;
                out.range() = t.range();
                in_stream.write(out);
            }

    std::cout << "[DeepWave] Streamed " << n_batches_group_aligned << " batches\n";

    // -------------------------------------------------------------------------
    // Run full kernel (DATAFLOW pipeline)
    //   Each frame produces (IMG_LEN + 1) 32-bit words: [norm, pixel0..pixelN]
    // -------------------------------------------------------------------------
    std::cout << "[DeepWave] Starting pipeline execution...\n";
    auto start = high_resolution_clock::now();

    int last_frame_count = 0;
    auto last_time = start;

    while (true) {
        deepwaveaccel(
            in_stream,
            param_bp_stream,
            param_db_stream,
            out_stream,
            goer_cfg,
            debl_cfg,
            status_gz,
            status_cc,
            status_bp,
            status_db
        );

        int frames_done = (int)(out_stream.size() / (IMG_LEN + 1));

        if (frames_done > last_frame_count) {
            auto now = high_resolution_clock::now();
            auto delta   = duration_cast<milliseconds>(now - last_time).count() / 1000.0;
            auto elapsed = duration_cast<milliseconds>(now - start).count() / 1000.0;

            std::cout << "  [Frame " << frames_done
                    << "/" << expected_frames
                    << "]  dt = " << std::fixed << std::setprecision(2)
                    << delta << " s  (total = "
                    << std::setprecision(2) << elapsed << " s)\n";

            last_time = now;
            last_frame_count = frames_done;
        }

        if (frames_done >= ((max_frames < 0)
                            ? expected_frames
                            : std::min(expected_frames, max_frames)))
            break;
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
        // First word per frame = norm
        out_axis_t w_norm_axis = out_stream.read();
        word_t w_norm = w_norm_axis.data;
        norm_sum_t nv;
        nv.range() = w_norm.range(norm_sum_t::width - 1, 0);
        norms.push_back(nv);

        // Then IMG_LEN pixel words
        for (int i = 0; i < IMG_LEN; ++i) {
            img_t pix;
            out_axis_t w_pix_axis = out_stream.read();
            word_t w_pix = w_pix_axis.data;
            pix.range() = w_pix.range(img_t::width - 1, 0);
            image_out.push_back(pix);
        }
    }

    std::cout << "[DeepWave] Collected " << image_out.size()
            << " pixels (" << (image_out.size() / IMG_LEN) << " frames)\n";
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
        for (size_t f = 0; f < norms.size(); ++f)
            csv_norm << f << "," << norms[f].to_double() << "\n";
        csv_norm.close();
        std::cout << "[DeepWave] Wrote norms to " << norms_out << "\n";
    }

    std::cout << "Done.\n";
    return 0;

}