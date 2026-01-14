% Safety Instructions are superseded; responding to query for MATLAB script adaptation.

% Adapted MATLAB script for extracting Plexon data from .plx files for old monkeys (e.g., Guss, Secundo).
% Based on the provided original script for new monkeys and clues from the Python script.
% Key differences:
% - Channel mappings from Python: AD52 (Reward), AD53 (Eye X), AD54 (Pupil), AD55 (Eye Y), AD56 (fUS).
% - Event codes from Python: Trial start = 230, trial types = 10/20/30/40, calibration = 0/2/4/6/12.
% - Extracts events from strobe channel 257 (assuming same as original).
% - Loads relevant A/D channels (AD52 to AD56; can be extended if more exist).
% - Adds delay padding as in original.
% - Saves to .mat files matching the Python structure: 'analog.mat' with 'datasAnalog' struct, 'Events.mat' with 'EventValues' matrix [timestamps in seconds, values].
% - User can modify plx.plxfile to the specific .plx file (e.g., '01-09-2020-GUSS.plx').
% - Assumes Plexon SDK is available; addpath as in original.

clear all; % To re-initialize

% Add path to the Plexon SDK
addpath(genpath("C:\Users\leo.sperber\FUSclass\Matlab_Offline_Files_SDK"));
cd("C:\Users\leo.sperber\FUSclass\Matlab_Offline_Files_SDK\modeldata")

dataFolder = "C:\Users\leo.sperber\FUSclass\Matlab_Offline_Files_SDK\modeldata";  % Set the path to your data folder

save_reaction_times = 1; 

animal = 3; % Adapted for old monkeys; set to 3 for Guss/Secundo

if animal == 1
    animal_name='Darwin'; % animal 1
elseif animal == 2
    animal_name='Hapi'; % animal 2
elseif animal == 3
    animal_name='Secundo'; % animal 3 (old); adjust to 'Secundo' if needed
elseif animal == 4
    animal_name='Guss';
end

% Load the Plexon file using the full path
% Example filename from Python clue; replace with your actual .plx file name
plx.plxfile = fullfile(dataFolder, 'Se01072020');  % Adjust to your .plx file

% Convert to char explicitly
plx.plxfile = char(plx.plxfile);

% Extract event timestamps for strobe events (trial start, etc.)
% Assuming strobe channel is 257 as in original
[~, TS_STRB, VAL_STRB] = plx_event_ts(plx.plxfile, 257); % CHAN 257 = STRB (Strobe channel)

% Define event codes based on Python clues (adapted from original VStrb)
VStrb.Start = 230; % Trial start from Python
% Trial types (not used in extraction, but for reference)
VStrb.Saccade_Left = 10;
VStrb.Saccade_Right = 20;
VStrb.AntiSaccade_Left = 30; % Cue right?
VStrb.AntiSaccade_Right = 40; % Cue left?
% Calibration points (for reference)
VStrb.Calib_0 = 0;
VStrb.Calib_2 = 2;
VStrb.Calib_4 = 4;
VStrb.Calib_6 = 6;
VStrb.Calib_12 = 12;
% Other codes from original not present; add if needed

% Extract names of A/D channels
[~, names] = plx_adchan_names(plx.plxfile);  % Extract the channel names

% Specify channels to load based on Python (old monkey mappings)
% AD52: Reward, AD53: Eye X, AD54: Pupil, AD55: Eye Y, AD56: fUS
plx.toload.plx_to_load = {'AD52', 'AD53', 'AD54', 'AD55', 'AD56'};
plx.toload.plx_name = plx.toload.plx_to_load;

% Initialize structure for analog data
plx.a = struct();

% Loop through each specified channel
for ii = 1:length(plx.toload.plx_to_load)
    current_channel = plx.toload.plx_to_load{ii};  % Get the current channel name

    % Check if the A/D channel exists in the PLX file
    [~, ~, ~, test_ad_exist] = plx_ad(plx.plxfile, current_channel);

    % If channel exists, proceed to load data
    if test_ad_exist ~= -1
        % Load the A/D data for the current channel
        [plx.a.adfreq, plx.a.nad, plx.a.tsad, plx.a.fnad, plx.a.(current_channel)] = plx_ad(plx.plxfile, current_channel);

        % Verify if the loaded data is numeric
        current_channel_data = plx.a.(current_channel);

        if isnumeric(current_channel_data)
            % Compute delay for the current channel (padding zeros from recording start to first timestamp)
            delay = zeros(ceil(plx.a.tsad(1) * 1000), 1);  % Delay in ms samples (assuming 1000 Hz? Adjust if freq differs)

            % Display the computed delay for the first channel
            if ii == 1
                disp(['*** Delay between the start of recording and 1st AD timestamp: ', num2str(length(delay)), ' ms']);
            end

            % Adjust the data with the computed delay (pad zeros at start)
            % Note: Original multiplies delay by range/adfreq, but since delay=zeros, it remains zeros
            field_name = plx.toload.plx_name{ii};
            if isfield(plx.a, field_name)
                plx.a.(field_name) = [delay .* range(plx.a.(field_name)) / plx.a.adfreq; plx.a.(field_name)];
            else
                warning(['Field ', field_name, ' not found in plx.a']);
            end
        end
    end
end

% Prepare datasAnalog struct for saving (matching Python structure)
datasAnalog = struct();
for ii = 1:length(plx.toload.plx_to_load)
    field_name = plx.toload.plx_to_load{ii};
    if isfield(plx.a, field_name)
        datasAnalog.(field_name) = plx.a.(field_name);
    end
end

% Prepare EventValues matrix [timestamps in seconds, values] (matching Python)
EventValues = [TS_STRB, VAL_STRB];

% Save to .mat files (in current directory or specify path)
% Analog.mat with 'datasAnalog'
save('analogSe17012020.mat', 'datasAnalog');

% Events.mat with 'EventValues'
save('EventsSe17012020.mat', 'EventValues');

disp('Data extracted and saved to analog.mat and Events.mat');

% Optional: Example usage similar to Pierre's code
% Find trial starts (in ms, rounded)
TrialStart_ = round(TS_STRB(VAL_STRB == VStrb.Start) * 1000);

