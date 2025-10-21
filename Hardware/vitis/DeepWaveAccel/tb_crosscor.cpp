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
    hls::stream<norm_sum_t>   norm_stream;

    // -------------------------------------------------------------------------
    // Load Goertzel results
    // -------------------------------------------------------------------------
    std::string file_in = std::string(OUTPUT_DIR) + "/goertzel_sim.csv";
    std::ifstream csv_in(file_in);
    if (!csv_in.is_open()) {
        std::cerr << "Failed to open input file: " << file_in << std::endl;
        return 1;
    }

    std::string header;
    std::getline(csv_in, header); // skip header

    int batch, channel;
    double re, im;
    std::vector<std::tuple<int,int,double,double>> rows;
    int max_ch = -1, max_batch = -1;

    while (csv_in >> batch) {
        char comma;
        csv_in >> comma >> channel >> comma >> re >> comma >> im;
        rows.emplace_back(batch, channel, re, im);
        if (channel > max_ch)  max_ch  = channel;
        if (batch   > max_batch) max_batch = batch;
    }
    csv_in.close();

    int N_CH = max_ch + 1;
    int N_BATCH = max_batch + 1;
    std::vector<std::vector<DFTc_t>> goertzel_out(N_CH, std::vector<DFTc_t>(N_BATCH));

    for (auto &r : rows) {
        int b  = std::get<0>(r);
        int ch = std::get<1>(r);
        re = std::get<2>(r);
        im = std::get<3>(r);
        goertzel_out[ch][b] = DFTc_t((DFT_t)re, (DFT_t)im);
    }

    std::cout << "Loaded " << N_BATCH << " batches from " << file_in << std::endl;

    // -------------------------------------------------------------------------
    // Stream input (each batch = 1 frame)
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
        + N_ELEM             // norm sum
        + NPAIR              // upper triangle output (no diagonals)
        + N_ELEM * N_ELEM;   // clear

    const int total_cycles = N_BATCH * cycles_per_frame;
    for (int i = 0; i < total_cycles; ++i)
        crosscor(in_stream, out_stream, norm_stream);

    // -------------------------------------------------------------------------
    // Collect outputs
    // Each GROUP_FRAMES input frames → one correlation matrix
    // -------------------------------------------------------------------------
    const int matrices = N_BATCH / GROUP_FRAMES;
    std::vector<std::vector<DFTc_t>> upper_out(matrices, std::vector<DFTc_t>(NPAIR));
    std::vector<norm_sum_t> norms(matrices);

    for (int m = 0; m < matrices; ++m) {
        // Only upper-triangle outputs
        for (int p = 0; p < NPAIR; ++p) {
            if (!out_stream.empty()) {
                AxisWordDFTc w = out_stream.read();
                upper_out[m][p] = DFTc_t(w.re, w.im);
            } else {
                std::cerr << "Missing upper data" << std::endl;
            }
        }

        if (!norm_stream.empty())
            norms[m] = norm_stream.read();
        else
            std::cerr << "Missing norm data" << std::endl;
    }

    // -------------------------------------------------------------------------
    // Write results to CSV
    // -------------------------------------------------------------------------
    std::string file_out = std::string(OUTPUT_DIR) + "/crosscor_sim.csv";
    std::ofstream csv_out(file_out);
    if (!csv_out.is_open()) {
        std::cerr << "Failed to open output file: " << file_out << std::endl;
        return 1;
    }

    csv_out << "matrix,index,real,imag\n";

    for (int m = 0; m < matrices; ++m) {
        for (int p = 0; p < NPAIR; ++p) {
            auto &v = upper_out[m][p];
            csv_out << m << "," << p << ","
                    << v.real().to_double() << ","
                    << v.imag().to_double() << "\n";
        }
    }
    csv_out.close();

    std::cout << "Wrote " << matrices
              << " matrices (upper only) to \"" << file_out << "\"" << std::endl;

    // Norms
    std::string norm_file = std::string(OUTPUT_DIR) + "/crosscor_norm_sim.csv";
    std::ofstream csv_norm(norm_file);
    if (!csv_norm.is_open()) {
        std::cerr << "Failed to open " << norm_file << std::endl;
    } else {
        csv_norm << "matrix,norm\n";
        for (int i = 0; i < matrices; ++i)
            csv_norm << i << "," << norms[i].to_double() << "\n";
        csv_norm.close();
        std::cout << "Wrote " << matrices
                  << " norm values to \"" << norm_file << "\"" << std::endl;
    }

    std::cout << "Done." << std::endl;
    return 0;
}
