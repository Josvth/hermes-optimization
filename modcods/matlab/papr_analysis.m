close all
clear all
clc

%% Setup the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 16);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["ID", "order", "rate", "Label", "isend", "ibend", "eta", "m1", "m2", "m3", "gamma1", "gamma2", "phi1", "phi2", "phi3", "papr"];
opts.VariableTypes = ["double", "categorical", "categorical", "string", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["Label", "papr"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["order", "rate", "Label", "papr"], "EmptyFieldRule", "auto");

% Import the data
dvbs2 = readtable("dvbs2.csv", opts);

%%
papr = zeros(1, height(dvbs2));

Nsym = 10000;

Nspan = 6;           % Filter span in symbol durations
beta = 0.35;         % Roll-off factor
sampsPerSym = 8;    % Upsampling factor

rctFilt = comm.RaisedCosineTransmitFilter(...
  'Shape',                  'Normal', ...
  'RolloffFactor',          beta, ...
  'FilterSpanInSymbols',    Nspan, ...
  'OutputSamplesPerSymbol', sampsPerSym);

% Normalize to obtain maximum filter tap value of 1
b = coeffs(rctFilt);
rctFilt.Gain = 1/max(b.Numerator);

for i = 1:height(dvbs2)
    modcod = dvbs2(i,:);
    
    % Load constellation mapping info
    M = [modcod.m1, modcod.m2, modcod.m3];
    gamma = [modcod.gamma1, modcod.gamma2];
    phi = [modcod.phi1, modcod.phi2, modcod.phi3] * pi / 180 - 1e-9;
            
    x = randi([0 sum(M)-1],Nsym,1);
    
    % Calculate and plot constellation
    if n_radii == 1
        y = pskmod(x, M, phi);
    else    
        y = apskmod(x, M, radii, phi);
    end

    % Calculate PAPR
    yfilt = rctFilt([y; zeros(Nsym/2,1)]);
    amp = abs(yfilt);
    peak = max(amp);
    avg = rms(amp);

    papr(i) = peak^2 / avg^2;
     
    Np = 1000;
    [f, xi] = ksdensity(amp, 'NumPoints', 1000);
    
    xiu = xi;
    xil = -1*xi;
    xi = -xi(end):xi(2)-xi(1):xi(end);

    fu = 0.5*(f / max(f));
    fl = fliplr(fu);
    
    f = interp1(xiu, fu, xi, 'linear', 0) + interp1(xil, fu, xi, 'linear', 0);
    
    if 1 
        figure('color', 'w')
        scatter(real(y), imag(y), 'x', 'r');
        title(modcod.Label)
        axis square
        grid on
        xlim([-1 1] * 1.5);
        ylim([-1 1] * 1.5);

        % Plot pdf
        figure('color', 'w')
        plot(xi, f);
    end
    
    
%     % Plot histogram
%     figure('color', 'w')
%     [N, edges] = histcounts(amp, 1000);
%     plot(edges(1:end-1), 20*log10(N))
    
end
