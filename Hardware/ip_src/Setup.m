clear all;
hdlsetuptoolpath('ToolName', 'Xilinx Vivado', 'ToolPath', 'C:\Xilinx\Vivado\2024.1\bin\vivado.bat');
fifo_buffer_in_size = 64; % Should be more than max_processor_delay
fifo_buffer_out_size = 10 * fifo_buffer_in_size; % Larger size to support all channels as output
max_processor_delay = 24; % This increases the maximum number of extra data pushes after ready has gone low. Useful when adding delays in the processor

%% Settings

% AXI4 Stream

% Parameters file
model_file = "../../Simulation/model_parameters_D1-5_freq0.mat";

% Wave file
wav_file = "../../Simulation/FRIDA/FRIDA/recordings/20160908/data_pyramic/segmented/two_speakers/1-5.wav";

load(model_file)

% Frequency of interest
ff = 1666.66;

% Cross correlation
% Group size
group_size = 9;
norm_gain_floor = 1;

% Num of power iterations
power_iters = 10;


% Deblurring
image_seed = zeros(1,n_px);

% Fixed point types
fixp_input_ws = 12;
fixp_dsp_ws = 18;
fixp_goertzel_int_bits = 4; % max 16. Goertzel IIR filter requires additional integer bits
fixp_eig_ws = 24;
fixp_eig_frac = fixp_input_ws+1;
fixp_eig_rec_ws = 24;
fixp_eig_rec_frac = fixp_eig_rec_ws - 1;

fixp_bpp_ws = 22;
fixp_bpp_frac = 24;

fixp_dbl_ws = fixp_dsp_ws;
fixp_dbl_frac = fixp_dsp_ws - 2;

fixp_lapmult_ws = fixp_dbl_ws;
fixp_lapmult_frac = fixp_dbl_frac;

fixp_chebacc_ws = fixp_dsp_ws + 4;
fixp_chebacc_frac = fixp_dbl_frac + 4;


%% FGPA setup

% Top level entity
model_name = 'Model';

fpga_clock = uint64(100e6); % Clock frequency of the FPGA (in Hz)
fs_bus=double(fpga_clock);

% Tool config
hdlset_param(model_name,'SynthesisTool','Xilinx Vivado')

% Configure delay blocks
delays = find_system(strcat(model_name,'/DUT_Test'), 'BlockType', 'Delay');

for i = 1:numel(delays)
    hdlset_param(delays{i}, 'UseRAM', 'on');
end


%% Simulink settings
h = simulink.sampletimecolors.Palette("myColors");
h.DiscreteSampleTimeColors = parula(7);
simulink.sampletimecolors.applyPalette(h);

%% Preprocessing

% Read the input files
[data_in, fs_in] = audioread(wav_file, 'native');

% Convert parameter ints to doubles
n_ch = double(n_ch);
n_px = double(n_px);
n_px_fi = fi(n_px,0,ceil(log2(n_px)));
n_layer = double(n_layer);
k = double(k);

% Goertzel
nffloat = 10 * fs_in / ff;
nf = 200; % Frame length
fr = fs_in / nf;
bin = round(ff / fr);
ff_actual = bin * fr;
bins = [bin, bin - 1]; % DeepWave reference bins
bin_cnt = size(bins,2);

omega_0 = 2 * pi * bins / nf;
cos_omega = 2 * cos(omega_0);
exp_omega = exp(1j * omega_0);

goertzel_fixp = fixdt(1,fixp_input_ws+fixp_goertzel_int_bits+2,fixp_input_ws+2-1);
cos_omega_fixp = fi(cos_omega, goertzel_fixp);
exp_omega_fixp = fi(exp_omega, goertzel_fixp);

hann_fixp = fixdt(0,fixp_input_ws,fixp_input_ws);
hann_table = 0.5*(1-cos(2*pi*(0:(nf/2-1))/(nf-1)));
hann_table_fixp = fi(hann_table, hann_fixp);

% Cross-cor
power_norm_scaling_factor = 1/sqrt(2);
power_norm_scaling_factor_fixp = fi(power_norm_scaling_factor, 0, 18);

% Backprojection
bpp_scaling_bits = 11;
b_scaled = b.*sqrt(2^bpp_scaling_bits);
b_fixp = fi(b_scaled, 1, 14);
tau_scaled = tau.*2^bpp_scaling_bits;
tau_fixp = fi(tau_scaled, 1, 13);

% For optimized method - not in use currently
[J, K] = find(triu(ones(n_ch), 1));  % upper triangle indices, j<k
J_ROM = J(:);  % row vector
K_ROM = K(:);

% Deblurring
beta_retanh = 1 / tanh(1.0);
beta_retanh_fixp = fi(beta_retanh,0,fixp_dbl_ws,fixp_dbl_frac);
image_seed_fixp = fi(image_seed,0,fixp_dbl_ws,fixp_dbl_frac);
theta_cor = beta_retanh * theta; % Correct for the removal of the activation gain
theta_fixp = fi(theta_cor,1,fixp_dbl_ws, fixp_dbl_frac);

% Laplacian
lap_diags = laplacian(:,2:end);
lap_offsets = laplacian(2:end,1);
lap_main = laplacian(1, 2); % Lap main diag is constant
lap_rest = laplacian(2:end, 2:end);
lap_rest_neg = -lap_rest;
diag_count = size(lap_offsets,1);

lap_offsets_fi = fi(lap_offsets, 0, ceil(log2(max(lap_offsets))));
lap_main_fi = fi(lap_main,0,fixp_lapmult_frac+ceil(log2(lap_main)),fixp_lapmult_frac);
lap_rest_fi = fi(lap_rest_neg,0,fixp_lapmult_frac+ceil(log2(max(lap_rest_neg(:)))),fixp_lapmult_frac); % lap_rest is non-positive, so invert and make unsigned
diag_count_fi = fi(diag_count, 0, ceil(log2(diag_count)));

%% Normalization divider LUT replacement - needs further investigation 
% if it's actually a noticeable improvement in speed-area-power

% Parameters
N_lut = 128;  % Number of LUT points

% Define fixed-point types
T_in  = fixdt(0,24,13);  % UFix24 Q13 input
T_out = fixdt(0,24,23);  % UFix24 Q23 output

% Input range (as double for computation)
x_min = 0.5;
x_max = double(fi(2^24-1, T_in));  % Max UFix24 with 13 fraction bits

% Logarithmic spacing for breakpoints
x_points_double = x_min * (x_max/x_min).^((0:N_lut-1)/(N_lut-1));

% Create fixed-point input breakpoints
lut_norm_x = fi(x_points_double, T_in);

% Preallocate LUT
lut_norm_y = fi(zeros(1,N_lut), T_out);

% Compute 1/x in native fixed-point
for i = 1:N_lut
    lut_norm_y(i) = fi(1 / double(lut_norm_x(i)), T_out);  % MATLAB handles Q23
end

%% HDL coder block setup
% These only need to be performed once, but I keep them here in case
% something needs to be changed later on.

% Find all delays in the CARFAC system that are dependent on 'nsec', i.e.
% are probably ideal subjects for RAM mapping

% model = 'Model/AxiStreamWrapper/Processor/ProcessSamples/CARFAC_tm_en/';
% delayblocks = find_system(model, ... 
%                             'regexp','on','LookUnderMasks', 'all','BlockType','Delay','DelayLength','nsec')

% delayblocks = find_system('Model/AxiStreamWrapper/Processor/ProcessSamples/CARFAC_tm_en/', ... 
%                             'regexp','on','BlockType','Delay','DelayLength','^(?!.*nsec).*$')

% Set the 'useRAM' property to On
% BE CAREFUL, SOME DELAYS ARE BETTER LEFT WITH UseRAM OFF (SMALL DELAYS)
% for ii=1:length(delayblocks)
%     hdlset_param(delayblocks{ii}, 'UseRAM', 'On')
% end

% Unfortunately we can't search for multiply-add block type (doesn't exist)
%, so we use the name instead.
% multiplyadds = find_system(model,'LookUnderMasks','all','regexp','on','Name','Multiply-Add');
% for ii=1:length(multiplyadds)
%     hdlset_param(multiplyadds{ii},'PipelineDepth','0');
% end

%%

cheb_in = out.image_in.Data;
cheb_out = out.image_out.Data;

function y = chebyshev_conv(L, x, theta)
%CHEBYSHEV_CONV Compute y = sum_{k=0}^K theta_k * z_k
%   where z_0 = x,
%         z_1 = L * x,
%         z_k = 2 * L * z_{k-1} - z_{k-2} for k >= 2
%
% Parameters:
%   L     - Sparse normalized Laplacian matrix (sparse matrix)
%   x     - Input vector/image
%   theta - Chebyshev coefficients [theta_0, theta_1, ..., theta_K]
%
% Returns:
%   y     - Output vector/image

    K = length(theta) - 1;

    z_k_minus_two = x;
    y = theta(1) * z_k_minus_two;  % MATLAB is 1-based indexing

    if K == 0
        return;
    end

    z_k_minus_one = L * x;
    y = y + theta(2) * z_k_minus_one;

    for k = 3:(K + 1)
        z_k = 2 * L * z_k_minus_one - z_k_minus_two;
        y = y + theta(k) * z_k;
        z_k_minus_two = z_k_minus_one;
        z_k_minus_one = z_k;
    end
end

function L_full = banded_to_full(L_banded)
%BANDED_TO_FULL Converts a symmetric banded matrix to full square matrix
% L_banded: [offsets, bands], size = (num_diags x N+1)
% Returns:
%   L_full: N x N full symmetric matrix

    [num_diags, N_plus_1] = size(L_banded);
    N = N_plus_1 - 1;
    L_full = zeros(N, N);
    
    offsets = L_banded(:, 1);
    bands = L_banded(:, 2:end);
    
    for i = 1:num_diags
        offset = offsets(i);
        band = bands(i, :).';  % Make column vector
        
        valid_len = N - offset;
        if valid_len <= 0
            continue;
        end
        
        idx_row = (1:valid_len).';
        idx_col = (1+offset:N).';
        
        % Upper triangular
        L_full(sub2ind([N, N], idx_row, idx_col)) = band(idx_col);
        
        % Symmetric lower part
        if offset > 0
            L_full(sub2ind([N, N], idx_col, idx_row)) = band(idx_col);
        else
            % Main diagonal
            L_full(sub2ind([N, N], idx_row, idx_row)) = band(idx_row);
        end
    end
end

lap_full = banded_to_full(laplacian);

cheb_out_mat = chebyshev_conv(lap_full, cheb_in',theta)';
cheb_out_mat = cheb_out_mat(1:end-1,:);

cheb_out = cheb_out(2:end,:);
max(abs(cheb_out - cheb_out_mat)./cheb_out_mat,[],"All")

writematrix(cheb_in, 'cheb_in.csv');

%% Export model output to python 

% Extract signals from the Simulink output
deblurred       = out.deblurred_fixp.Data;   % or out.deblurred.Data
deblurred_valid = out.status_fixp.valid.Data;
norm_value       = squeeze(out.status_fixp.norm.Data);
norm_floor_hit   = squeeze(out.status_fixp.norm_floor_hit.Data);

% Keep only valid samples
deblurred = deblurred(deblurred_valid==1,:);
norm_value = norm_value(deblurred_valid==1,:);
norm_floor_hit = norm_floor_hit(deblurred_valid==1,:);

% Cast to double
norm_value = double(norm_value);
deblurred = double(deblurred);

% Scale deblurred from fixedpoint to reference scaling
deblurred = deblurred * 2^-11 * beta_retanh;

% Save all to the same MAT file
save("deblurred.mat", "deblurred", "norm_value", "norm_floor_hit");

%% Export model parameters to VitisHLS
PARAM_DIR = fullfile('..','vitis','DeepWaveAccel','parameters');

% ------------------------------------------------
% Export b (steering vectors)
% ------------------------------------------------
fid_b = fopen(fullfile(PARAM_DIR,'b_vectors.csv'), 'w');
fprintf(fid_b, 'pixel,elem,real,imag\n');
for pix = 1:n_px
    for elem = 1:n_ch
        val = b_scaled(elem, pix);
        fprintf(fid_b, '%d,%d,%.10f,%.10f\n', pix-1, elem-1, real(val), imag(val));
    end
end
fclose(fid_b);

% ------------------------------------------------
% Export tau (per-pixel correction)
% ------------------------------------------------
y_diag = mean(abs(b_scaled(:)).^2)*sqrt(2);
tau_adj = tau_scaled - y_diag;
fid_tau = fopen(fullfile(PARAM_DIR,'tau.csv'), 'w');
fprintf(fid_tau, 'tau\n');
for i = 1:n_px
    fprintf(fid_tau, '%.10f\n', tau_adj(i));
end
fclose(fid_tau);

% ------------------------------------------------
% Export Laplacian (main + off-diagonals)
% ------------------------------------------------
fid_lap = fopen(fullfile(PARAM_DIR,'laplacian.csv'), 'w');
fprintf(fid_lap, 'lap_value\n');
fprintf(fid_lap, '%.12f\n', lap_main);
[ND, IMG_LEN] = size(lap_rest_neg);
for d = 1:ND
    for i = 1:IMG_LEN
        fprintf(fid_lap, '%.12f\n', lap_rest_neg(d,i));
    end
end
fclose(fid_lap);

% ------------------------------------------------
% Export Laplacian offsets and Theta coefficients
% ------------------------------------------------
writematrix(lap_offsets(:)', fullfile(PARAM_DIR,'lap_offsets.csv'));
writematrix(theta_cor(:)', fullfile(PARAM_DIR,'theta.csv'));


%% Ref outputs

goertzel_ref = squeeze(out.goertzel_ref.Data(1,:,out.goertzel_ref_valid.Data))';
crosscor_ref = permute(squeeze(out.crosscor_ref.Data(:,:,out.crosscor_ref_valid.Data)),[3 1 2]);
bpp_ref = squeeze(out.bpp_ref.Data(out.bpp_ref_valid.Data,:));
deblurred_ref = squeeze(out.deblurred_ref.Data(out.deblurred_ref_valid.Data,:));

save("refs.mat", "goertzel_ref", "crosscor_ref", "bpp_ref", "deblurred_ref");

%% FixP to binary
idx=2;
var=cos_omega_fixp; 
rawstr = var(idx);
rawstr = rawstr.bin;
pointIndex = var.WordLength - var.FractionLength;
str1 = rawstr(1:pointIndex);
str2 = '.';
str3 = rawstr(pointIndex+1:end);
outstr = ['0b' str1 str2 str3];
disp(outstr)