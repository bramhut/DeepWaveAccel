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

% Frequency of interest
ff = 1666.66;

% Cross correlation
% Group size
group_size = 9;

% Num of power iterations
power_iters = 10;

% Read these files
[data_in, fs_in] = audioread(wav_file, 'native');
load(model_file)

%% FGPA setup

fpga_clock = uint64(100e6); % Clock frequency of the FPGA (in Hz)
fs_bus=double(fpga_clock);

% WordLengths (Note to self: DSP's are 18 bit)
wl_sample = 24;

%% Simulink tools
pltObj = simulink.sampletimecolors.Palette("SampleTimeColors");
pltObj.DiscreteSampleTimeColors = parula(6);
simulink.sampletimecolors.applyPalette(pltObj, 'UserDefault',true')

%% Preprocessing

% Convert parameter ints to doubles
n_ch = double(n_ch);
n_px = double(n_px);
n_layer = double(n_layer);


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
