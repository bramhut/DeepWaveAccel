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

% Num of power iterations
power_iters = 10;


% Deblurring
image_seed = zeros(1,n_px);



%% FGPA setup

fpga_clock = uint64(100e6); % Clock frequency of the FPGA (in Hz)
fs_bus=double(fpga_clock);

% WordLengths (Note to self: DSP's are 18 bit)
wl_sample = 24;

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
n_layer = double(n_layer);
k = double(k);

% Split laplacian
laplacian_diags_offsets = int16(laplacian(:,1));
laplacian_diags = laplacian(:, 2:end);
n_laplacian_diags = size(laplacian_diags,1);


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

% Activation function
beta_retanh = 1 / tanh(1.0);

%% Input plotting


% Create time vector
time = (0:size(data_in, 1)-1) / fs_in;

% Plot the first column of d_raw
figure;
plot(time, data_in(:, 1));
xlabel('Time (s)');
ylabel('d_{raw} Column 1');
title('Plot of d_{raw} Column 1 vs Time');
grid on;

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