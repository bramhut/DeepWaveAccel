#include "tb_base.hpp"
#include "deblur.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

// -------------------------------------------------------------
// Deblur testbench (cycle-accurate simulation)
// -------------------------------------------------------------
int tb_deblur() {
    hls::stream<AxisWordImg> bp_stream;
    hls::stream<lap_t>       lap_stream;
    hls::stream<AxisWordImg> img_stream;

    deblur_config cfg;
    cfg.n_layers = 5;
    cfg.K = 22;

    // ---------------------------------------------------------
    // Load Laplacian coefficients (CSV)
    //  - first row: main diagonal scalar
    //  - next ND*IMG_LEN rows: off-diagonal values
    // ---------------------------------------------------------
    std::string lap_file = std::string(PARAM_DIR) + "/laplacian.csv";
    std::ifstream lap_in(lap_file);
    if (!lap_in.is_open()) {
        std::cerr << "Failed to open Laplacian file: " << lap_file << std::endl;
        return 1;
    }

    std::string header;
    std::getline(lap_in, header); // skip header

    double val;
    bool first = true;
    int d = 0, i = 0;
    while (lap_in >> val) {
        if (first) {
            lap_stream.write((lap_t)val); // main diagonal
            first = false;
            std::cout << "[Deblur] Loaded laplacian main: " << (lap_t)val << std::endl;
        } else {
            lap_stream.write((lap_t)val);
            ++i;
            if (i == IMG_LEN) {
                i = 0;
                ++d;
            }
        }
    }
    lap_in.close();
    std::cout << "[Deblur] Loaded Laplacian: main + " << d << " off-diagonals" << std::endl;

    // ---------------------------------------------------------
    // Load Laplacian offsets (AXI-Lite config)
    // ---------------------------------------------------------
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
            cfg.lap_off[k] = (idx_t)std::stod(token);
        } catch (...) {
            std::cerr << "Invalid offset format in file!" << std::endl;
            return 1;
        }
    }
    off_in.close();

    std::cout << "[Deblur] Loaded " << ND << " Laplacian offsets" << std::endl;

    // ---------------------------------------------------------
    // Load θ coefficients (AXI-Lite config)
    // ---------------------------------------------------------
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
            cfg.theta[k] = (theta_t)std::stod(token);
        } catch (...) {
            std::cerr << "Invalid theta coefficient format in file!" << std::endl;
            return 1;
        }
    }
    th_in.close();
    std::cout << "[Deblur] Loaded theta coefficients (K=" << (int)cfg.K << ")" << std::endl;


    // ---------------------------------------------------------
    // Load backprojection input image
    // ---------------------------------------------------------
    std::string bp_file = std::string(OUTPUT_DIR) + "/backproj_sim.csv";
    std::ifstream bp_in(bp_file);
    if (!bp_in.is_open()) {
        std::cerr << "Failed to open backprojection file: " << bp_file << std::endl;
        return 1;
    }

    std::getline(bp_in, header); // skip header
    int frame, pixel;
    double value;
    int count = 0;
    while (bp_in >> frame) {
        char comma;
        bp_in >> comma >> pixel >> comma >> value;
        AxisWordImg word;
        word.data = (img_t)value;
        word.user = (pixel == 0);
        word.last = (pixel == IMG_LEN - 1);
        bp_stream.write(word);
        ++count;
    }
    bp_in.close();

    auto frame_count = count / IMG_LEN;
    std::cout << "[Deblur] Loaded " << count << " backprojection pixels (" 
              << (count / IMG_LEN) << " frames)" << std::endl;

    // ---------------------------------------------------------
    // Run deblur kernel
    // ---------------------------------------------------------
    std::cout << "[Deblur] Starting kernel execution..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    const int cycles_per_frame =
        1 + ND * IMG_LEN + IMG_LEN * (int)cfg.n_layers * (int)cfg.K;  // safe upper bound

    while(img_stream.size() < frame_count*IMG_LEN)
        deblur(bp_stream, lap_stream, img_stream, cfg);

    if (!bp_stream.empty()){
        std::cerr << "[Deblur] Did not process all frames! Exiting early. " << bp_stream.size() << " pixels left (" << bp_stream.size()/IMG_LEN << " frames)" << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "[Deblur] Kernel finished in " << dur / 1000.0 << " s" << std::endl;

    // ---------------------------------------------------------
    // Collect output image(s)
    // ---------------------------------------------------------
    std::vector<img_t> output;
    while (!img_stream.empty()) {
        AxisWordImg ow = img_stream.read();
        output.push_back(ow.data);
    }

    std::cout << "[Deblur] Collected " << output.size()
              << " pixels (" << output.size() / IMG_LEN << " frames)" << std::endl;

    // ---------------------------------------------------------
    // Write results to CSV
    // ---------------------------------------------------------
    std::string file_out = std::string(OUTPUT_DIR) + "/deblur_sim.csv";
    std::ofstream csv_out(file_out);
    if (!csv_out.is_open()) {
        std::cerr << "Failed to open output file: " << file_out << std::endl;
        return 1;
    }

    csv_out << "frame,pixel,value\n";
    for (size_t i = 0; i < output.size(); ++i) {
        int frame = i / IMG_LEN;
        int pix = i % IMG_LEN;
        double scaled_out = output[i].to_double() / (2048.0 * std::tanh(1.0));
        csv_out << frame << "," << pix << "," << scaled_out << "\n";
    }
    csv_out.close();

    std::cout << "[Deblur] Wrote results to " << file_out << std::endl;
    std::cout << "✅ Done." << std::endl;
    return 0;
}
