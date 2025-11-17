% load dataset
real_folder1 = 'data_TorontoScenes/CP_images';
sun397_folders = {'SUN397_IMAGES/dataset/bedroom', 'SUN397_IMAGES/dataset/cockpit', 'SUN397_IMAGES/dataset/forest', 'SUN397_IMAGES/dataset/highway'};
synthetic_folder = 'stableDiffusion_dataset_rev2/IMAGES';

resize_dim = [256, 256];

% torontoscenes 
real_images1 = dir(fullfile(real_folder1, '*.jpg'));
num_real_images1 = length(real_images1);

% SUN397
sun397_images = [];
for k = 1:length(sun397_folders)
    sun_folder = sun397_folders{k};
    images = dir(fullfile(sun_folder, '*.jpg'));
    sun397_images = [sun397_images; images];
end
num_sun397_images = length(sun397_images);


num_real_images = num_real_images1 + num_sun397_images;
real_data = zeros([resize_dim, num_real_images]);


for i = 1:num_real_images1
    img = imread(fullfile(real_folder1, real_images1(i).name));
    img_gray = rgb2gray(im2double(img));
    img_resized = imresize(img_gray, resize_dim);
    real_data(:, :, i) = img_resized;
end


for i = 1:num_sun397_images
    img = imread(fullfile(sun397_folders{ceil(i/num_sun397_images)}, sun397_images(i).name));
    img_gray = rgb2gray(im2double(img));
    img_resized = imresize(img_gray, resize_dim);
    real_data(:, :, num_real_images1 + i) = img_resized;
end

% Load Synthetic Images
synthetic_images = dir(fullfile(synthetic_folder, '*.png'));
num_synthetic_images = length(synthetic_images);
synthetic_data = zeros([resize_dim, num_synthetic_images]);

for i = 1:num_synthetic_images
    img = imread(fullfile(synthetic_folder, synthetic_images(i).name));
    img_gray = rgb2gray(im2double(img));
    img_resized = imresize(img_gray, resize_dim);
    synthetic_data(:, :, i) = img_resized;
end

% compute DCT (dct2)
real_dct = zeros(size(real_data));
for i = 1:num_real_images
    real_dct(:, :, i) = dct2(real_data(:, :, i));
end

synthetic_dct = zeros(size(synthetic_data));
for i = 1:num_synthetic_images
    synthetic_dct(:, :, i) = dct2(synthetic_data(:, :, i));
end

real_avg_spectrum = mean(abs(real_dct), 3); % take avg to highlight patterns
synthetic_avg_spectrum = mean(abs(synthetic_dct), 3);

% plot 1: Average DCT Spectrum - Real Images
log_real_spectrum = log10(real_avg_spectrum + 1e-6); % enhance small variations
figure;
imagesc(log_real_spectrum);
colormap('jet');
colorbar;
axis image;
title('Average DCT Spectrum - Real Images');
xlabel('Frequency (kx)');
ylabel('Frequency (ky)');

% plot 2: Average DCT Spectrum - Synthetic Images
log_synthetic_spectrum = log10(synthetic_avg_spectrum + 1e-6);
figure;
imagesc(log_synthetic_spectrum);
colormap('jet');
colorbar;
axis image;
title('Average DCT Spectrum - Synthetic Images');
xlabel('Frequency (kx)');
ylabel('Frequency (ky)');

% plot 3: Difference Spectrum - Real vs Synthetic
difference_spectrum = abs(real_avg_spectrum - synthetic_avg_spectrum);
log_difference_spectrum = log10(difference_spectrum + 1e-6);

figure;
imagesc(log_difference_spectrum);
colormap('jet');
colorbar;
axis image;
title('Difference Spectrum: Real vs Synthetic');
xlabel('Frequency (kx)');
ylabel('Frequency (ky)');

% Additional: Histogram of DCT Coefficients

% Flatten DCT coefficients for histograms
real_dct_flat = reshape(abs(real_dct), [], 1);
synthetic_dct_flat = reshape(abs(synthetic_dct), [], 1);

% Remove extremely high values for better visualization (optional, e.g., clip at 99.9 percentile)
real_dct_flat = real_dct_flat(real_dct_flat <= prctile(real_dct_flat, 99.9));
synthetic_dct_flat = synthetic_dct_flat(synthetic_dct_flat <= prctile(synthetic_dct_flat, 99.9));

% plot 4: Histogram of Real Images DCT Coefficients
figure;
histogram(real_dct_flat, 100, 'Normalization', 'probability');
title('Histogram of DCT Coefficients - Real Images');
xlabel('DCT Coefficient Magnitude');
ylabel('Probability');

% plot 5: Histogram of Synthetic Images DCT Coefficients
figure;
histogram(synthetic_dct_flat, 100, 'Normalization', 'probability');
title('Histogram of DCT Coefficients - Synthetic Images');
xlabel('DCT Coefficient Magnitude');
ylabel('Probability');

% Frequency Band Analysis

% Define low, mid, and high-frequency ranges
low_freq = 1:85; % Low-frequency range (example: first 1/3 of frequencies)
mid_freq = 86:170; % Mid-frequency range (example: second 1/3 of frequencies)
high_freq = 171:256; % High-frequency range (example: last 1/3 of frequencies)

% Extract average coefficients for each range
real_low_freq = mean(real_avg_spectrum(low_freq, low_freq), 'all');
real_mid_freq = mean(real_avg_spectrum(mid_freq, mid_freq), 'all');
real_high_freq = mean(real_avg_spectrum(high_freq, high_freq), 'all');

synthetic_low_freq = mean(synthetic_avg_spectrum(low_freq, low_freq), 'all');
synthetic_mid_freq = mean(synthetic_avg_spectrum(mid_freq, mid_freq), 'all');
synthetic_high_freq = mean(synthetic_avg_spectrum(high_freq, high_freq), 'all');

% Plot bar graph to compare frequency bands
figure;
bar([real_low_freq, synthetic_low_freq; real_mid_freq, synthetic_mid_freq; real_high_freq, synthetic_high_freq]);
xticks(1:3);
xticklabels({'Low Frequency', 'Mid Frequency', 'High Frequency'});
legend('Real Images', 'Synthetic Images');
title('Comparison of DCT Coefficients by Frequency Bands');
xlabel('Frequency Band');
ylabel('Average DCT Coefficient Magnitude');
