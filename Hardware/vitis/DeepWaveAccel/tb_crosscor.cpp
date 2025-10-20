#include "tb_base.hpp"
#include "crosscor.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <complex>
#include <string>

int tb_crosscor() {
    hls::stream<AxisWordDFTc> in_stream;
    hls::stream<AxisWordDFTc> out_stream;
    hls::stream<norm_sum_t> norm_stream;

    // -------------------------------------------------------------------------
    // Load Goertzel results (input to crosscor)
    // -------------------------------------------------------------------------
    std::string file_in = std::string(OUTPUT_DIR) + "/goertzel_sim.csv";
    std::ifstream csv_in(file_in);
    if (!csv_in.is_open()) {
        std::cerr << "Failed to open input file: " << file_in << "\n";
        return 1;
    }

    std::string header;
    std::getline(csv_in, header); // skip header

    int channel, batch;
    double re, im;
    std::vector<std::vector<DFTc_t>> goertzel_out;

    // Determine dimensions automatically
    int max_ch = -1, max_batch = -1;
    std::vector<std::tuple<int, int, double, double>> rows;

    while (csv_in >> batch) {
        char comma;
        csv_in >> comma >> channel >> comma >> re >> comma >> im;
        rows.push_back({batch, channel, re, im});
        if (channel > max_ch) max_ch = channel;
        if (batch > max_batch) max_batch = batch;
    }
    csv_in.close();

    int N_CH = max_ch + 1;
    int N_BATCH = max_batch + 1;
    goertzel_out.resize(N_CH, std::vector<DFTc_t>(N_BATCH));

    for (auto &r : rows) {
        int b  = std::get<0>(r);
        int ch = std::get<1>(r);
        re = std::get<2>(r);
        im = std::get<3>(r);
        goertzel_out[ch][b] = DFTc_t((DFT_t)re, (DFT_t)im);
    }

    std::cout << "Loaded " << N_BATCH << " batches from " << file_in << "\n";

    // -------------------------------------------------------------------------
    // Stream input to crosscor
    // (each batch is a frame: vector<DFTc_t> of 48 elements)
    // -------------------------------------------------------------------------
    for (int b = 0; b < N_BATCH; ++b) {
        for (int ch = 0; ch < N_ELEM; ++ch) {
            bool first = (ch == 0);
            bool last  = (ch == N_ELEM - 1);
            in_stream.write(AxisWordDFTc(goertzel_out[ch][b], last, first));
        }
    }

    // -------------------------------------------------------------------------
    // Run CrossCor kernel
    // -------------------------------------------------------------------------
    const int cycles_per_frame = 
      N_ELEM * N_ELEM   // correlation
    + N_ELEM             // diagonal sum
    + N_ELEM * N_ELEM    // output
    + N_ELEM * N_ELEM;   // clear

    const int total_cycles = N_BATCH * cycles_per_frame;

    for (int i = 0; i < total_cycles; ++i) {
        crosscor(in_stream, out_stream, norm_stream);
    }

    // -------------------------------------------------------------------------
    // Collect and store outputs
    // Each 9 input frames (GROUP_FRAMES) produce 1 correlation matrix (48×48)
    // -------------------------------------------------------------------------
    const int matrices = N_BATCH / GROUP_FRAMES;
    std::vector<std::vector<std::vector<DFTc_t>>> corr_out(
        matrices, std::vector<std::vector<DFTc_t>>(N_ELEM, std::vector<DFTc_t>(N_ELEM))
    );
    std::vector<norm_sum_t> norms(matrices); // Vector of norms

    for (int m = 0; m < matrices; ++m) {
        for (int i = 0; i < N_ELEM; ++i) {
            for (int j = 0; j < N_ELEM; ++j) {
                if (!out_stream.empty()) {
                    AxisWordDFTc w = out_stream.read();
                    corr_out[m][i][j] = DFTc_t(w.re, w.im);
                }
                else {
                    std::cerr << "This should not happen! No data to read from out_stream...\n";  
                }
            }
        }
        if (!norm_stream.empty()){
            norms[m] = norm_stream.read();
        } else {
            std::cerr << "This should not happen! No data to read from norm_stream...\n";  
        }
    }

    // -------------------------------------------------------------------------
    // Write results to CSV
    // -------------------------------------------------------------------------
    std::string file_out = std::string(OUTPUT_DIR) + "/crosscor_sim.csv";
    std::ofstream csv_out(file_out);
    if (!csv_out.is_open()) {
        std::cerr << "Failed to open output CSV file: " << file_out << "\n";
        return 1;
    }

    csv_out << "matrix,row,col,real,imag\n";

    for (int m = 0; m < matrices; ++m) {
        for (int i = 0; i < N_ELEM; ++i) {
            for (int j = 0; j < N_ELEM; ++j) {
                auto &v = corr_out[m][i][j];
                csv_out << m << "," << i << "," << j << ","
                        << v.real() << ","
                        << v.imag() << "\n";
            }
        }
    }

    csv_out.close();
    std::cout << "Wrote " << matrices << " correlation matrices (" << N_ELEM << "x" << N_ELEM
              << ") to \"" << file_out << "\"\n";

    // Norms
    std::string norm_file = std::string(OUTPUT_DIR) + "/crosscor_norm_sim.csv";
    std::ofstream csv_norm(norm_file);
    if (!csv_norm.is_open()) {
        std::cerr << "Failed to open " << norm_file << "\n";
    } else {
        csv_norm << "matrix,norm\n";
        for (int i = 0; i < matrices; ++i)
            csv_norm << i << "," << norms[i] << "\n";
        csv_norm.close();
        std::cout << "Wrote " << matrices
                << " norm values to \"" << norm_file << "\"\n";
    }

    std::cout << "Done.\n";

    return 0;
}
