#include "tb_base.hpp"
#include "backproj.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <string>
#include <chrono>

using namespace std::chrono;

// -----------------------------------------------------------------------------
// Backprojection testbench
// -----------------------------------------------------------------------------
int tb_backproj() {
    hls::stream<AxisWordDFTc> corr_stream;
    hls::stream<b_t>          b_stream;
    hls::stream<tau_t>        tau_stream;
    hls::stream<AxisWordImg>  img_stream;

    // -------------------------------------------------------------------------
    // Load correlation data (crosscor_sim.csv)
    // -------------------------------------------------------------------------
    std::string file_in = std::string(OUTPUT_DIR) + "/crosscor_sim.csv";
    std::ifstream csv_in(file_in);
    if (!csv_in.is_open()) {
        std::cerr << "Failed to open input file: " << file_in << std::endl;
        return 1;
    }

    std::string header;
    std::getline(csv_in, header); // skip header

    int matrix, index;
    double re, im;
    std::vector<std::tuple<int,int,double,double>> rows;

    // CSV format: matrix,index,real,imag
    while (csv_in >> matrix) {
        char comma;
        csv_in >> comma >> index >> comma >> re >> comma >> im;
        rows.emplace_back(matrix, index, re, im);
    }
    csv_in.close();

    // Determine number of correlation matrices (frames)
    int N_MAT = std::get<0>(rows.back()) + 1;

    std::cout << "[Backproj] Loaded " << N_MAT
              << " correlation matrices (upper only) from " << file_in << std::endl;

    // -------------------------------------------------------------------------
    // Load steering vectors (b_stream) ONCE
    // -------------------------------------------------------------------------
    std::string b_file = std::string(PARAM_DIR) + "/b_vectors.csv";
    std::ifstream b_in(b_file);
    if (!b_in.is_open()) {
        std::cerr << "Failed to open steering vector file: " << b_file << std::endl;
        return 1;
    }

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

    std::cout << "[Backproj] Loaded " << b_count
              << " steering elements (streamed once)" << std::endl;

    // -------------------------------------------------------------------------
    // Load tau values ONCE (already compensated for diagonal)
    // -------------------------------------------------------------------------
    std::string tau_file = std::string(PARAM_DIR) + "/tau.csv";
    std::ifstream tau_in(tau_file);
    if (!tau_in.is_open()) {
        std::cerr << "Failed to open tau file: " << tau_file << std::endl;
        return 1;
    }

    std::getline(tau_in, header); // skip header

    double tau_val;
    int tau_count = 0;
    while (tau_in >> tau_val) {
        tau_stream.write((tau_t)tau_val);
        ++tau_count;
    }
    tau_in.close();

    std::cout << "[Backproj] Loaded " << tau_count
              << " tau values (streamed once)" << std::endl;

    // -------------------------------------------------------------------------
    // Stream correlation data (upper triangle only, already in correct order)
    // -------------------------------------------------------------------------
    for (auto &r : rows) {
        double re = std::get<2>(r);
        double im = std::get<3>(r);
        corr_stream.write(AxisWordDFTc((DFT_t)re, (DFT_t)im));
    }

    std::cout << "[Backproj] Streaming " << rows.size()
            << " correlation entries (upper only, ordered)" << std::endl;

    // -------------------------------------------------------------------------
    // Run Backprojection kernel
    // -------------------------------------------------------------------------
    const int LOAD_TAU_CYCLES   = IMG_LEN;                 // tau: IMG_LEN entries
    const int LOAD_B_CYCLES     = N_ELEM * IMG_LEN;        // b: N_ELEM * IMG_LEN entries
    const int LOAD_SIGMA_FRAME  = NPAIR;                   // upper only (no diag)
    const int PER_PIXEL_CYCLES  = NPAIR + 1;               // COMPUTE_UP + OUTPUT
    const int PIPELINE_SLACK    = 32;

    const int TOTAL_CYCLES =
        N_MAT * (LOAD_SIGMA_FRAME + IMG_LEN * PER_PIXEL_CYCLES) +
        PIPELINE_SLACK;

    std::cout << "[Backproj] Loading b and tau parameters to BRAM..." << std::endl;

    // Load parameters
    for (int i = 0; i < (LOAD_TAU_CYCLES + LOAD_B_CYCLES); ++i)
        backprojection(corr_stream, b_stream, tau_stream, img_stream);

    if (!b_stream.empty() || !tau_stream.empty()) {
        std::cerr << "[Backproj] ERROR: Loading of b and tau NOT completely finished" << std::endl;
    } else {
        std::cout << "[Backproj] Loading of b and tau finished" << std::endl;
    }

    std::cout << "[Backproj] Starting kernel execution with a total of "
              << TOTAL_CYCLES << " cycles" << std::endl;

    auto start = high_resolution_clock::now();

    for (int i = 0; i < TOTAL_CYCLES; ++i)
        backprojection(corr_stream, b_stream, tau_stream, img_stream);

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    std::cout << "[Backproj] Finished kernel execution in "
              << duration.count() / 1000.0 << " s" << std::endl;

    // -------------------------------------------------------------------------
    // Collect output images
    // -------------------------------------------------------------------------
    std::vector<bp_out_t> image_vals;
    while (!img_stream.empty()) {
        AxisWordImg out = img_stream.read();
        image_vals.push_back(out.data);
    }

    std::cout << "[Backproj] Collected " << image_vals.size()
              << " pixel results (" << (image_vals.size() / IMG_LEN)
              << " frames)" << std::endl;

    // -------------------------------------------------------------------------
    // Write results to CSV
    // -------------------------------------------------------------------------
    std::string file_out = std::string(OUTPUT_DIR) + "/backproj_sim.csv";
    std::ofstream csv_out(file_out);
    if (!csv_out.is_open()) {
        std::cerr << "Failed to open output file: " << file_out << std::endl;
        return 1;
    }

    csv_out << "frame,pixel,value\n";
    for (size_t i = 0; i < image_vals.size(); ++i) {
        int frame = i / IMG_LEN;
        int pixel = i % IMG_LEN;
        csv_out << frame << "," << pixel << ","
                << image_vals[i].to_double() << "\n";
    }
    csv_out.close();

    std::cout << "Wrote " << image_vals.size()
              << " pixel results to \"" << file_out << "\"" << std::endl;

    std::cout << "Done." << std::endl;
    return 0;
}
