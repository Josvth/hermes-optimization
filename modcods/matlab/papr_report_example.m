close all
clear all
clc

addpath('../scripts')
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

%% Plot vs alpha
modcod_id = 25;
alphas = [0.20, 0.25, 0.35];

Nsym = 10000;

xx = cell(1, length(alphas));
yy = cell(1, length(alphas));
yy_filt = cell(1, length(alphas));
ppapr = cell(1, length(alphas));
dds_pdf = cell(1, length(alphas));

for i = 1:length(alphas)
    
    modcod = dvbs2(modcod_id,:);
    alpha = alphas(i);
    
    [xx{i}, yy{i}, yy_filt{i}, ppapr{i}, dds_pdf{i}] = estimate_modulation_parameters(modcod, alpha, Nsym);
    
end

% Plotting
figure('color', 'w')

for i = 1:length(alphas)
    y = yy{i};
    y_filt = yy_filt{i};
    papr = ppapr{i};
    ds_pdf = dds_pdf{i};   
    plot(ds_pdf.b, ds_pdf.pdf);
    hold on
    
    leg{i} = sprintf('\\alpha = %0.2f', alphas(i));
    
end

grid on
xlim([-2 2]);
xlabel('b [V/V_{max}]');

ylabel('pdf');
legend(leg);
PlotUtil.apply_report_formatting()

%% Plot vs modcod
modcod_ids = [25:27];
alpha = 0.25;

Nsym = 10000;

xx = cell(1, length(alphas));
yy = cell(1, length(alphas));
yy_filt = cell(1, length(alphas));
ppapr = cell(1, length(alphas));
dds_pdf = cell(1, length(alphas));

for i = 1:length(modcod_ids)
    
    modcod = dvbs2(modcod_ids(i),:);
    
    [xx{i}, yy{i}, yy_filt{i}, ppapr{i}, dds_pdf{i}] = estimate_modulation_parameters(modcod, alpha, Nsym);
    
end

% Plotting
figure('color', 'w')

for i = 1:length(alphas)
    y = yy{i};
    y_filt = yy_filt{i};
    papr = ppapr{i};
    ds_pdf = dds_pdf{i};   
    plot(ds_pdf.b, ds_pdf.pdf);
    hold on
    
    leg{i} = sprintf('ACM=%d', modcod_ids(i));
    
end

grid on
xlim([-2 2]);
xlabel('b [V/R_{max}]');
ylabel('pdf');
legend(leg);
PlotUtil.apply_report_formatting()

%% Plot vs modcod
modcod_ids = [0,11,17,23] + 1;
alpha = 0.25;

Nsym = 10000;

xx = cell(1, length(alphas));
yy = cell(1, length(alphas));
yy_filt = cell(1, length(alphas));
ppapr = cell(1, length(alphas));
dds_pdf = cell(1, length(alphas));

for i = 1:length(modcod_ids)
    
    modcod = dvbs2(modcod_ids(i),:);
    
    [xx{i}, yy{i}, yy_filt{i}, ppapr{i}, dds_pdf{i}] = estimate_modulation_parameters(modcod, alpha, Nsym);
    
end

% Plotting
figure('color', 'w')

for i = 1:length(modcod_ids)
    y = yy{i};
    y_filt = yy_filt{i};
    papr = ppapr{i};
    ds_pdf = dds_pdf{i};   
    plot(ds_pdf.b, ds_pdf.pdf);
    hold on
    
    leg{i} = sprintf('%s', dvbs2.('Label')(modcod_ids(i)));
    
end

grid on
xlim([-2 2]);
xlabel('b [V/R_{max}]');
ylabel('pdf');
legend(leg);
PlotUtil.apply_report_formatting_single(1,1)
ax.Legend.FontSize = 8;

%%
function [x, y, y_filt, papr, ds_pdf] = estimate_modulation_parameters(modcod, alpha, Nsym)
    
    if nargin < 3
        Nsym = 1000;
    end
    
    % Load constellation mapping info
    M = [modcod.m1, modcod.m2, modcod.m3];
    gamma = [modcod.gamma1, modcod.gamma2];
    phi = [modcod.phi1, modcod.phi2, modcod.phi3] * pi / 180 - 1e-9;
    
    % Build constellation
    [M, phi, radii] = process_apsk_parameters(M, gamma, phi);
    
    n_radii = sum(M > 0);
    
    x = randi([0 sum(M)-1],Nsym,1);
    
    % Calculate and plot constellation
    if n_radii == 1
        y = pskmod(x, M, phi);
    else    
        y = apskmod(x, M, radii, phi);
    end

    if 0
        % Filter
        fc = 30e9;
        fs = 4*fc;
        Rs = 30e6;      % 30 Msym/s

        y_filt = nyquist_filter(y, alpha, fs/Rs);

        % Modulate 
        t = (0:length(y_filt)-1)*1/fs;
        y_rf = real(y_filt.*exp(-1j*2*fc*t'));

        % Calculate PAPR
        papr = calculate_papr(y_rf);

        % Calculate double sided pdf
        [b, pdf] = double_sided_pdf_hist(y_rf);
        ds_pdf.pdf = pdf;
        ds_pdf.b = b;
        
    else
        % Filter
        y_filt = nyquist_filter(y, alpha, 200);

        % Calculate PAPR
        papr = calculate_papr(y_filt);

        % Calculate double sided pdf
        [b, pdf] = double_sided_pdf_hist(y_filt);
        ds_pdf.pdf = pdf;
        ds_pdf.b = b;
    end
            
end

function [M, phi, radii] = process_apsk_parameters(M, gamma, phi)

    % Reduce to actual constellation
    n_radii = sum(M > 0);
    M = M(1:n_radii);
    gamma = gamma(1:n_radii-1);
    if n_radii == 1
        radii = 1;
    else
        radii = zeros(1, n_radii);
        radii(end) = 1;
        radii(1) = 1/gamma(end);
        for j = 1:n_radii-2
            radii(j+1) = radii(1) * gamma(end-j);
        end
    end
    phi = phi(1:n_radii);

end

function x_filt = nyquist_filter(x, alpha, samplesPerSym)

    Nspan = 10;                  % Filter span in symbol durations
    
    if nargin < 3
        samplesPerSym = 10;
    end
    
    rctFilt = comm.RaisedCosineTransmitFilter(...
      'Shape',                  'Square root', ...
      'RolloffFactor',          alpha, ...
      'FilterSpanInSymbols',    Nspan, ...
      'OutputSamplesPerSymbol', samplesPerSym);

    % Normalize to obtain maximum filter tap value of 1
    b = coeffs(rctFilt);
    rctFilt.Gain = 1/max(b.Numerator);
    
    x_filt = rctFilt(x);
    x_filt = x_filt(rctFilt.FilterSpanInSymbols*4:end);
end

function papr = calculate_papr(x)
    
    if isreal(x)
        peak = max(abs(x));
        avg = rms(x);
        papr = peak^2 / avg^2;
    else
        peak = max(abs(x));
        avg = rms(abs(x));
        papr = 2 * peak^2 / avg^2;
    end
    
end

function [b, ds_pdf] = double_sided_pdf_hist(x)

    if nargin < 2
        num_points = 201;
    end
    
    if isreal(x)
        mag = x;
    else
        mag = [abs(x), -1*abs(x)];
    end
    
    [ds_pdf, b] = histcounts(mag, num_points);
    ds_pdf = ds_pdf / sum(ds_pdf);
    b = b(1:end-1) + diff(b)/2;
    
end

function [b, ds_pdf] = double_sided_pdf_mirror(x, num_points)

    if nargin < 2
        num_points = 201;
    end

    amp = [abs(x); -1*abs(x)];
    
    [ds_pdf, b] = ksdensity(amp, 'NumPoints', num_points);
    ds_pdf = ds_pdf / sum(ds_pdf);
     
end

function [b, ds_pdf] = double_sided_pdf(x, num_points)

    if nargin < 2
        num_points = 1001;
    end

    amp = abs(x);
    
    [pdf, u] = ksdensity(amp, 'NumPoints', num_points);
    
    uu = u;
    ul = -1*uu;
    b = -u(end):u(2)-u(1):u(end);

    pdfu = 0.5*(pdf);
    pdfl = fliplr(pdfu.');
    
    ds_pdf = max(interp1(uu, pdfu, b, 'linear', 'extrap'), 0) + max(interp1(ul, pdfl, b,'linear', 'extrap'), 0);
    ds_pdf = ds_pdf / sum(ds_pdf);
    
end

