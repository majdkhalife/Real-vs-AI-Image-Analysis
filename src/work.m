%HERE WE LOAD OUR IMAGES
AiImage = imread("AI images/img12.png");
RealImage = imread("Real Images/img5.png");

%CONVERT TO GRAYSCALE
grayAI = rgb2gray(AiImage);
grayRealImage = rgb2gray(RealImage);

%APPLY CANNY
edgesAI = edge(grayAI, 'canny');
edgesReal = edge(grayRealImage, 'canny');

%FIND OUR CONTOURS
[contoursAi, img1] = bwboundaries(edgesAI, 'noholes');
[contoursReal, img2] = bwboundaries(edgesReal, 'noholes');

%Show me contours
figure;
subplot(1, 2, 1);
imshow(AiImage);
hold on;
title('Contours - Image 1 (AI)');
for k = 1:length(contoursAi)
    plot(contoursAi{k}(:, 2), contoursAi{k}(:, 1), 'r', 'LineWidth', 1);
end

subplot(1, 2, 2);
imshow(RealImage);
hold on;
title('Contours - Image 2 (Natural)');
for k = 1:length(contoursReal)
    plot(contoursReal{k}(:, 2), contoursReal{k}(:, 1), 'r', 'LineWidth', 1);
end

%ANALYZING CONTOURS(HELPER) AND SHOWING RESULTS
metrics1 = analyzeContours(contoursAi, grayAI, AiImage);
metrics2 = analyzeContours(contoursReal, grayRealImage, RealImage);

disp('Metrics for Image 1 (AI-generated):');
disp(metrics1);

disp('Metrics for Image 2 (Natural):');
disp(metrics2);

compareMetrics(metrics1, metrics2);

%ANALYZE CONTOURS AND COMPUTE METRICS
function metrics = analyzeContours(contours, grayImage, colorImage)
    numContours = length(contours);
    contourLengths = zeros(numContours, 1);
    avgCurvatures = zeros(numContours, 1);
    orientations = [];
    textureFeatures = [];
    
    %vars for color analysis
    colorMeans = [];
    colorStds = [];
    colorSkews = [];

    for i = 1:numContours
        %get a contour
        contour = contours{i};
        x = contour(:, 2); 
        y = contour(:, 1);

        %calculate its length using euclidean distance cuz hes the
        %goat(euclid)
        lengths = sqrt(diff(x).^2 + diff(y).^2);
        contourLengths(i) = sum(lengths);

        %contour curvature(found this formula on stackoverflow, also goats)
        dx = gradient(x);
        dy = gradient(y);
        d2x = gradient(dx);
        d2y = gradient(dy);
        curvature = abs(d2x .* dy - d2y .* dx) ./ ((dx.^2 + dy.^2).^(3/2));
        if ~isempty(curvature)
            avgCurvatures(i) = mean(curvature);
        else
            avgCurvatures(i) = 0;
        end

        %now we can get our contour orientation, we use tan to get the
        %angle of the vector xy in 2d plane.
        angles = atan2d(diff(y), diff(x)); % angles in degrees
        orientations = [orientations; angles];
    end

    %use harris for junctuon detection
    corners = detectHarrisFeatures(grayImage);
    junctionCount = corners.Count;

    %do texture analysis with GLCM
    offsets = [0 1; -1 1; -1 0; -1 -1];
    glcm = graycomatrix(grayImage, 'Offset', offsets);
    stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
    textureFeatures.Contrast = mean(stats.Contrast);
    textureFeatures.Correlation = mean(stats.Correlation);
    textureFeatures.Energy = mean(stats.Energy);
    textureFeatures.Homogeneity = mean(stats.Homogeneity);

    %here we do color analsyis
    %get color moments
    for channel = 1:3
        colorChannel = colorImage(:, :, channel);
        colorMeans(channel) = mean(colorChannel(:));
        colorStds(channel) = std(double(colorChannel(:)));
        colorSkews(channel) = skewness(double(colorChannel(:)));
    end

    %use FFT to get PSD
    fftImage = fft2(double(grayImage));
    fftShifted = fftshift(fftImage);
    magnitudeSpectrum = abs(fftShifted);
    psd = magnitudeSpectrum.^2;
    psdMean = mean(psd(:));

    %Preppig for comparisons
    metrics.TotalContours = numContours;
    metrics.MeanContourLength = mean(contourLengths);
    metrics.MeanCurvature = mean(avgCurvatures);
    metrics.OrientationHistogram = histcounts(orientations, -180:20:180);
    metrics.JunctionCount = junctionCount;

    metrics.TextureContrast = textureFeatures.Contrast;
    metrics.TextureCorrelation = textureFeatures.Correlation;
    metrics.TextureEnergy = textureFeatures.Energy;
    metrics.TextureHomogeneity = textureFeatures.Homogeneity;

    metrics.ColorMean = colorMeans;
    metrics.ColorStd = colorStds;
    metrics.ColorSkewness = colorSkews;

    % Frequency features
    metrics.PSDMean = psdMean;
end

%this function will print the metrics for both, aiding us in coming to a
%conclusion


function compareMetrics(metrics1, metrics2)
    disp('Comparing Metrics between AI-generated and Natural Images:');

    %mean contour length
    fprintf('Mean Contour Length - AI: %.2f, Real: %.2f\n', metrics1.MeanContourLength, metrics2.MeanContourLength);

    %mean curvature
    fprintf('Mean Curvature - AI: %.2f, Real: %.2f\n', metrics1.MeanCurvature, metrics2.MeanCurvature);

    %junction counts
    fprintf('Junction Count - AI: %d, Real: %d\n', metrics1.JunctionCount, metrics2.JunctionCount);

    %texture features(GLCM)
    fprintf('Texture Contrast - AI: %.2f, Real: %.2f\n', metrics1.TextureContrast, metrics2.TextureContrast);
    fprintf('Texture Correlation - AI: %.2f, Real: %.2f\n', metrics1.TextureCorrelation, metrics2.TextureCorrelation);
    fprintf('Texture Energy - AI: %.2f, Real: %.2f\n', metrics1.TextureEnergy, metrics2.TextureEnergy);
    fprintf('Texture Homogeneity - AI: %.2f, Real: %.2f\n', metrics1.TextureHomogeneity, metrics2.TextureHomogeneity);

    %color features[moments]
    fprintf('Color Mean - AI: [%.2f %.2f %.2f], Real: [%.2f %.2f %.2f]\n', ...
        metrics1.ColorMean(1), metrics1.ColorMean(2), metrics1.ColorMean(3), ...
        metrics2.ColorMean(1), metrics2.ColorMean(2), metrics2.ColorMean(3));
    fprintf('Color Std Dev - AI: [%.2f %.2f %.2f], Real: [%.2f %.2f %.2f]\n', ...
        metrics1.ColorStd(1), metrics1.ColorStd(2), metrics1.ColorStd(3), ...
        metrics2.ColorStd(1), metrics2.ColorStd(2), metrics2.ColorStd(3));
    fprintf('Color Skewness - AI: [%.2f %.2f %.2f], Real: [%.2f %.2f %.2f]\n', ...
        metrics1.ColorSkewness(1), metrics1.ColorSkewness(2), metrics1.ColorSkewness(3), ...
        metrics2.ColorSkewness(1), metrics2.ColorSkewness(2), metrics2.ColorSkewness(3));

    %freq features, thanks to FFT
    fprintf('Mean PSD - AI: %.2e, Real: %.2e\n', metrics1.PSDMean, metrics2.PSDMean);

end
