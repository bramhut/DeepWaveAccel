#include "tb_base.hpp"
#include "top.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <complex>

int tb_goertzel();
int tb_crosscor();
int tb_top();

int main() {
    tb_goertzel();
    // tb_crosscor();
    // tb_top();

    // using test_t = ap_fixed<3,1>;
    // using testc_t = complex<test_t>;
    // test_t re_in = 0.25, im_in = -0.5;
    // testc_t c_in = testc_t(re_in, im_in);
    // corr_accum_t test = std::conj((corr_accum_t)c_in) * (corr_accum_t)c_in;
    // auto test2 = std::conj(c_in) * c_in;
    // auto test3 = std::conj(c_in);
    // std::cout << "test: " << test << " test2: " << test2 << " test3: " << test3 << std::endl;

}

int tb_top() {
    hls::stream<AxisWordSampleIn> in_stream;
    hls::stream<AxisWordDFTc> out_stream;
    goertzel_config cfg;

    // Read WAV file
    const char* filename = "C:/Users/bramh/Documents/Github/DeepWaveAccel/Simulation/FRIDA/FRIDA/recordings/20160908/data_pyramic/segmented/two_speakers/1-5.wav";
    std::vector<int16_t> wav_samples;
    int channels, samplerate, frames;
    if(!read_wav_16bit(filename, wav_samples, channels, samplerate, frames)) {
        std::cerr << "Failed to read WAV file or unsupported format\n";
        return 1;
    }

    if (channels != N_ELEM){
        std::cerr << "Number of channels in input wave file does not match DeepWaveAccel setup\n";
        return 1;
    }

    // Prepare configuration
    goertzel_prepare_config(cfg, (double)samplerate, FF);

    std::cout << "Calculated Goertzel coefficients:\n";
    for (int b = 0; b < NBINS; b++) {
        std::cout << "  cos_omega[" << b << "] = " << cfg.COS_OMEGA[b].to_double()
                  << ", sin_omega[" << b << "] = " << cfg.SIN_OMEGA[b].to_double() << "\n";
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
        deepwaveaccel(in_stream, out_stream, cfg);
    }

    // Collect outputs into array [N_ELEM][n_batches]
    std::vector<std::vector<DFTc_t>> deepwave_out(N_ELEM, std::vector<DFTc_t>(n_batches));

    for(int b = 0; b < n_batches; ++b) {
        for(int ch = 0; ch < N_ELEM; ++ch) {
            if(!out_stream.empty()) {
                AxisWordDFTc in = out_stream.read();
                deepwave_out[ch][b] = std::complex<DFT_t>(in.re, in.im);
            }
        }
    }

    // Write results to CSV file
    std::string file_sim_out =  "/goertzel_sim.csv";
    std::ofstream csv(std::string(OUTPUT_DIR) + file_sim_out);
    if (!csv.is_open()) {
        std::cerr << "Failed to open output CSV file\n";
        return 1;
    }

    // Optional header
    csv << "channel,batch,real,imag\n";

    for (int ch = 0; ch < N_ELEM; ++ch) {
        for (int b = 0; b < n_batches; ++b) {
            auto& out = deepwave_out[ch][b];
            csv << ch << "," << b << ","
                << out.real().to_double() << ","
                << out.imag().to_double() << "\n";
        }
    }

    csv.close();
    std::cout << "Wrote " << N_ELEM * n_batches << " results to \"output" << file_sim_out << "\"\n";

    std::cout << "Done.\n";
    return 0;
}
