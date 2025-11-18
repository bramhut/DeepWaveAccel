#include "tb_base.hpp"
#include "top.hpp"
#include <dirent.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>


using namespace std::chrono;

int tb_deepwaveaccel_all();
int tb_deepwaveaccel();

// int tb_goertzel();
// int tb_crosscor();
// int tb_backproj();
// int tb_deblur();

int main() {
    std::cout.setf(std::ios::unitbuf);
    std::setvbuf(stdout, NULL, _IONBF, 0);

    tb_deepwaveaccel_all();
    // tb_deepwaveaccel();
    // tb_goertzel();
    // tb_crosscor();
    // tb_backproj();
    // tb_deblur();
}

struct DWResult {
    int frames;
    std::vector<img_t> image;
    std::vector<norm_out_t> norms;
};


bool load_parameters(
    std::vector<b_t>& b_ddr,
    std::vector<tau_t>& tau_ddr,
    std::vector<lap_t>& lap_ddr,
    std::vector<idx_t>& lap_offs,
    std::vector<theta_t>& theta_vals)
{
    std::string header, line;

    // ---------------------------------------------
    // b_vectors.csv
    // ---------------------------------------------
    {
        std::string b_file = std::string(PARAM_DIR) + "/b_vectors.csv";
        std::ifstream b_in(b_file);
        if (!b_in.is_open()) {
            std::cerr << "Failed to open steering vector file: " << b_file << "\n";
            return false;
        }

        std::getline(b_in, header);
        b_ddr.resize(IMG_LEN * N_ELEM);

        int pixel, elem;
        double bre, bim;
        int b_count = 0;

        while (b_in >> pixel) {
            char comma;
            b_in >> comma >> elem >> comma >> bre >> comma >> bim;
            b_ddr[pixel * N_ELEM + elem] = b_t((b_real_t)bre, (b_real_t)bim);
            ++b_count;
        }

        std::cout << "[DeepWave] Loaded " << b_count << " steering-vector entries\n";
    }

    // ---------------------------------------------
    // tau.csv
    // ---------------------------------------------
    {
        std::string tau_file = std::string(PARAM_DIR) + "/tau.csv";
        std::ifstream tau_in(tau_file);
        if (!tau_in.is_open()) {
            std::cerr << "Failed to open tau file: " << tau_file << "\n";
            return false;
        }

        std::getline(tau_in, header);
        tau_ddr.resize(IMG_LEN);

        double tau_val;
        int i = 0;
        while (tau_in >> tau_val && i < IMG_LEN)
            tau_ddr[i++] = (tau_t)tau_val;

        std::cout << "[DeepWave] Loaded " << tau_ddr.size() << " tau values\n";
    }

    // ---------------------------------------------
    // laplacian.csv
    // ---------------------------------------------
    {
        std::string lap_file = std::string(PARAM_DIR) + "/laplacian.csv";
        std::ifstream lap_in(lap_file);
        if (!lap_in.is_open()) {
            std::cerr << "Failed to open Laplacian file: " << lap_file << "\n";
            return false;
        }

        std::getline(lap_in, header);
        lap_ddr.clear();

        double v;
        while (lap_in >> v)
            lap_ddr.push_back((lap_t)v);

        std::cout << "[DeepWave] Loaded Laplacian: " << lap_ddr.size() << " entries\n";
    }

    // ---------------------------------------------
    // lap_offsets.csv
    // ---------------------------------------------
    {
        std::string offset_file = std::string(PARAM_DIR) + "/lap_offsets.csv";
        std::ifstream off_in(offset_file);
        if (!off_in.is_open()) {
            std::cerr << "Failed to open Laplacian offset file: " << offset_file << "\n";
            return false;
        }

        lap_offs.clear();
        if (std::getline(off_in, line)) {
            std::stringstream ss(line);
            std::string token;

            while (std::getline(ss, token, ',')) {
                try {
                    lap_offs.push_back((idx_t)std::stoi(token));
                } catch (...) {
                    std::cerr << "Invalid Laplacian offset: " << token << "\n";
                    return false;
                }
            }
        }

        if ((int)lap_offs.size() != ND) {
            std::cerr << "[DeepWave] Warning: Expected " << ND
                      << " Laplacian offsets but got " << lap_offs.size() << "\n";
        }

        std::cout << "[DeepWave] Loaded Laplacian offsets (" << lap_offs.size()
                  << " values)\n";
    }

    // ---------------------------------------------
    // theta.csv
    // ---------------------------------------------
    {
        std::string theta_file = std::string(PARAM_DIR) + "/theta.csv";
        std::ifstream th_in(theta_file);
        if (!th_in.is_open()) {
            std::cerr << "Failed to open theta file: " << theta_file << "\n";
            return false;
        }

        theta_vals.clear();
        if (std::getline(th_in, line)) {
            std::stringstream ss(line);
            std::string token;
            while (std::getline(ss, token, ',')) {
                try {
                    theta_vals.push_back((theta_t)std::stod(token));
                } catch (...) {
                    std::cerr << "Invalid theta coefficient: " << token << "\n";
                    return false;
                }
            }
        }

        std::cout << "[DeepWave] Loaded " << theta_vals.size()
                  << " theta coefficients (K=" << (theta_vals.size() - 1) << ")\n";
    }

    return true;
}

bool load_and_apply_parameters(
    hls::stream<word_t>& param_bp_stream,
    hls::stream<word_t>& param_db_stream,
    const goertzel_config& goer_cfg,
    const deblur_config&  debl_cfg)
{
    // --- Local storage of parameters ---
    std::vector<b_t>    b_ddr;
    std::vector<tau_t>  tau_ddr;
    std::vector<lap_t>  lap_ddr;
    std::vector<idx_t>  lap_offs;
    std::vector<theta_t> theta_vals;

    // 1) LOAD ALL PARAMETERS FROM CSV
    if (!load_parameters(b_ddr, tau_ddr, lap_ddr, lap_offs, theta_vals))
        return false;

    // 2) STREAM PARAMETERS INTO param_bp_stream / param_db_stream
    {
        word_t p;

        // --- Backprojection ---
        for (int pix = 0; pix < IMG_LEN; pix++) {
            for (int e = 0; e < N_ELEM; e++) {
                const b_t& b = b_ddr[pix*N_ELEM + e];
                p.range() = b.real().range(); param_bp_stream.write(p);
                p.range() = b.imag().range(); param_bp_stream.write(p);
            }
        }

        for (int pix = 0; pix < IMG_LEN; pix++) {
            p.range() = tau_ddr[pix].range();
            param_bp_stream.write(p);
        }

        // --- Deblur ---
        int K = (int)theta_vals.size() - 1;
        p = K;
        param_db_stream.write(p);

        for (int i = 0; i <= K; i++) {
            p.range() = theta_vals[i].range();
            param_db_stream.write(p);
        }

        for (int i = 0; i < ND; i++) {
            p = (unsigned)lap_offs[i];
            param_db_stream.write(p);
        }

        p.range() = lap_ddr[0].range();
        param_db_stream.write(p);

        int idx = 1;
        for (int d = 0; d < ND; d++)
            for (int i = 0; i < IMG_LEN; i++) {
                p.range() = lap_ddr[idx++].range();
                param_db_stream.write(p);
            }
    }

    std::cout << "[DeepWave] Parameters successfully loaded into the streams.\n";
    return true;
}

void write_binary_results(
    const std::string& wav_rel,
    const DWResult& R,
    std::ofstream& bin_px,
    std::ofstream& bin_nm,
    std::ofstream& index_csv,
    int& global_frame_counter)
{
    int start = global_frame_counter;
    int F = R.frames;

    index_csv << wav_rel << "," << start << "," << F << "\n";

    // write norms
    for (int f = 0; f < F; f++) {
        float fn = (float)R.norms[f].to_double();
        bin_nm.write((char*)&fn, sizeof(float));
    }

    // write pixels
    for (int f = 0; f < F; f++) {
        for (int p = 0; p < IMG_LEN; p++) {
            size_t idx = (size_t)f * IMG_LEN + p;
            float val = (float)(R.image[idx].to_double()
                                / (2048.0 * std::tanh(1.0)));
            bin_px.write((char*)&val, sizeof(float));
        }
    }

    global_frame_counter += F;
}


DWResult run_one_wav(
    const std::string& wav_path,
    const goertzel_config& goer_cfg,
    const deblur_config& debl_cfg,
    hls::stream<word_t>& param_bp_stream,
    hls::stream<word_t>& param_db_stream)
{
    DWResult R;
    R.frames = 0;

    // Extract simple file name from wav_path
    std::string file_name = wav_path.substr(wav_path.find_last_of("/\\") + 1);

    // --- LOAD WAV ---
    std::vector<int16_t> wav;
    int ch, sr, nsmp;
    if (!read_wav_16bit(wav_path, wav, ch, sr, nsmp)) {
        std::cerr << "[ERR] Cannot read WAV: " << wav_path << "\n";
        return R;
    }
    if (ch != N_ELEM) {
        std::cerr << "[ERR] Wrong channel count in " << wav_path << "\n";
        return R;
    }

    int n_batches = nsmp / N_WIN;
    int n_batches_aligned = (n_batches / GROUP_FRAMES) * GROUP_FRAMES;
    int frames = n_batches_aligned / GROUP_FRAMES;
    if (frames <= 0) return R;
    R.frames = frames;

    hls::stream<word_t>     in_stream;
    hls::stream<out_axis_t> out_stream;

    // --- STREAM WAV SAMPLES ---
    for (int b = 0; b < n_batches_aligned; b++)
        for (int c = 0; c < N_ELEM; c++)
            for (int n = 0; n < N_WIN; n++) {
                int idx = (b*N_WIN + n)*N_ELEM + c;
                sample_t t = 16.0 * double(wav[idx]) / 32768.0;
                word_t w; w.range() = t.range();
                in_stream.write(w);
            }

    // --- RUN KERNEL UNTIL ALL FRAMES PRODUCED ---

    status_gz_t sg; status_cc_t sc;
    status_bp_t sb; status_db_t sd;

    while (true) {
        deepwaveaccel(in_stream,
                      param_bp_stream, param_db_stream,
                      out_stream,
                      goer_cfg, debl_cfg,
                      sg, sc, sb, sd);

        int produced = out_stream.size() / (IMG_LEN + 1);
        if (produced >= frames)
            break;
    }

    // --- READ OUT ALL FRAMES ---
    R.image.reserve((size_t)frames * IMG_LEN);
    R.norms.reserve(frames);

    for (int f = 0; f < frames; f++) {
        out_axis_t wn = out_stream.read();
        norm_out_t nv;
        nv.range() = wn.data.range(norm_out_t::width-1,0);
        R.norms.push_back(nv);

        for (int i = 0; i < IMG_LEN; i++) {
            out_axis_t wp = out_stream.read();
            img_t ix;
            ix.range() = wp.data.range(img_t::width-1,0);
            R.image.push_back(ix);
        }
    }

    return R;
}



int tb_deepwaveaccel_all()
{
    std::cout << "[TB] Full FRIDA dataset\n";

    auto tb_start = high_resolution_clock::now();

    // --- Config ---
    goertzel_config goer_cfg;
    deblur_config   debl_cfg;
    debl_cfg.n_layers = 5;
    goertzel_prepare_config(goer_cfg, 16000.0, FF);

    // --- Stream parameters ONCE ---
    hls::stream<word_t> param_bp_stream;
    hls::stream<word_t> param_db_stream;

    if (!load_and_apply_parameters(param_bp_stream, param_db_stream, goer_cfg, debl_cfg))
        return 1;

    // --- Outputs ---
    std::string out_pixels = std::string(OUTPUT_DIR) + "/deepwave_pixels.bin";
    std::string out_norms  = std::string(OUTPUT_DIR) + "/deepwave_norms.bin";
    std::string out_index  = std::string(OUTPUT_DIR) + "/deepwave_index.csv";

    std::ofstream bin_px(out_pixels, std::ios::binary);
    std::ofstream bin_nm(out_norms,  std::ios::binary);

    std::ofstream index_csv(out_index);
    index_csv << "wav,start_frame,frames\n";

    int global_frame_counter = 0;

    // ---------------------------------------------------------
    // 1) Count total wav files in all folders (for progress)
    // ---------------------------------------------------------
    auto count_wav_files = [&](const std::string& folder) {
        int count = 0;
        std::string full = std::string(WAVE_DIR) + "/" + folder;
        DIR* dir = opendir(full.c_str());
        if (!dir) return 0;

        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr)
        {
            std::string n = entry->d_name;
            if (n.size() >= 4 && n.substr(n.size()-4) == ".wav")
                count++;
        }
        closedir(dir);
        return count;
    };

    int total_files = count_wav_files("one_speaker");
    // total_files +=    count_wav_files("two_speakers");
    // total_files +=    count_wav_files("three_speakers");

    if (total_files == 0) {
        std::cerr << "[ERROR] No .wav files found. Check WAVE_DIR.\n";
        return 1;
    }

    std::cout << "[INFO] Total WAV files to process: " << total_files << "\n";

    int processed_files = 0;

    // ---------------------------------------------------------
    // 2) Folder processing lambda with progress counter
    // ---------------------------------------------------------
    auto process_folder = [&](const std::string& folder) {
        std::string full = std::string(WAVE_DIR) + "/" + folder;
        std::cout << "\n[INFO] Processing folder: " << full << "\n";

        DIR* dir = opendir(full.c_str());
        if (!dir) {
            std::cerr << "[WARN] Cannot open folder: " << full << "\n";
            return;
        }

        struct dirent* entry;

        while ((entry = readdir(dir)) != nullptr) {
            std::string name = entry->d_name;

            if (name == "." || name == "..") continue;
            if (name.size() < 4 || name.substr(name.size() - 4) != ".wav") continue;

            processed_files++;

            std::string wav_rel  = folder + "/" + name;
            std::string wav_path = full  + "/"  + name;

            auto t0 = high_resolution_clock::now();

            DWResult R = run_one_wav(
                wav_path, goer_cfg, debl_cfg,
                param_bp_stream, param_db_stream
            );

            auto t1 = high_resolution_clock::now();

            if (R.frames <= 0) {
                std::cerr << "[ERROR] No frames produced for: " << wav_rel << "\n";
                continue;
            }

            write_binary_results(
                wav_rel, R,
                bin_px, bin_nm,
                index_csv,
                global_frame_counter
            );

            double dt_s  = duration_cast<seconds>(t1 - t0).count();
            double dt_min = dt_s / 60.0;
            double sec_per_frame = dt_s / R.frames;

            // --- PROGRESS: one-line summary ---
            std::cout << "[FILE "
                      << processed_files << "/" << total_files << "] "
                      << wav_rel
                      << " | Frames: " << R.frames
                      << " | Time: " << std::fixed << std::setprecision(1)
                      << dt_min << " min (" << dt_s << " s)"
                      << " | Avg: " << std::setprecision(2)
                      << sec_per_frame << " s/frame"
                      << "\n";
        }

        closedir(dir);
    };

    // ---------------------------------------------------------
    // 3) Run all folders
    // ---------------------------------------------------------
    process_folder("one_speaker");
    // process_folder("two_speakers");
    // process_folder("three_speakers");

    // ---------------------------------------------------------
    // 4) Total summary
    // ---------------------------------------------------------
    auto tb_end = high_resolution_clock::now();
    double tb_s = duration_cast<seconds>(tb_end - tb_start).count();
    double tb_min = tb_s / 60.0;

    std::cout << "\n[TB] Total frames across all files: "
              << global_frame_counter << "\n";

    std::cout << "[TB] Total processing time: "
              << std::fixed << std::setprecision(1)
              << tb_min << " minutes (" << tb_s << " seconds)\n";

    std::cout << "[TB] Output written to:\n"
              << "      " << out_pixels << "\n"
              << "      " << out_norms  << "\n"
              << "      " << out_index  << "\n";

    return 0;
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
    std::vector<norm_out_t> norms;
    norms.reserve(expected_frames);

    std::vector<img_t> image_out;
    image_out.reserve((size_t)expected_frames * IMG_LEN);

    while (!out_stream.empty()) {
        // First word per frame = norm
        out_axis_t w_norm_axis = out_stream.read();
        word_t w_norm = w_norm_axis.data;
        norm_out_t nv;
        nv.range() = w_norm.range(norm_out_t::width - 1, 0);
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