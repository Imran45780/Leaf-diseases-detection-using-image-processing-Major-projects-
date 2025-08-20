clc % Clear command window
close all % Close all figures
clear all % Clear all variables

% Load an image file using a dialog box
[filename, pathname] = uigetfile({'.';'.bmp';'.jpg';'*.gif'}, 'Pick a Leaf Image File');
I = imread([pathname,filename]); % Read the selected image
I = imresize(I,[256,256]); % Resize the image to 256x256 pixels
figure, imshow(I); title('Query Leaf Image'); % Display the input image

% Enhance contrast using histogram stretching
I = imadjust(I,stretchlim(I));
figure, imshow(I); title('Contrast Enhanced');

% Convert image to binary using Otsu thresholding method
I_Otsu = im2bw(I,graythresh(I)); % Threshold calculated as T = argmin( w0(T)*?0² + w1(T)*?1² )
I_HIS = rgb2hsi(I); % Convert RGB to HSI color space (Hue, Saturation, Intensity)

%% Extract Features using Color Space Transformation and Clustering
cform = makecform('srgb2lab'); % Create color space transformation structure from sRGB to Lab
lab_he = applycform(I,cform); % Apply Lab color transformation

ab = double(lab_he(:,:,2:3)); % Extract 'a' and 'b' channels for color-based segmentation
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2); % Reshape to 2D array for clustering

nColors = 3; % Number of clusters for K-means
% K-means clustering to segment image into color clusters (using squared Euclidean distance)
[cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean','Replicates',3);

% Reshape cluster indices back into image dimensions
pixel_labels = reshape(cluster_idx,nrows,ncols);

% Separate each cluster and create segmented images
segmented_images = cell(1,3);
rgb_label = repmat(pixel_labels,[1,1,3]); % Prepare for RGB image labeling
for k = 1:nColors
    colors = I;
    colors(rgb_label ~= k) = 0; % Mask out other clusters
    segmented_images{k} = colors; % Store segmented cluster image
end

% Display segmented images (clusters)
figure, subplot(3,1,1);imshow(segmented_images{1});title('Cluster 1');
subplot(3,1,2);imshow(segmented_images{2});title('Cluster 2');
subplot(3,1,3);imshow(segmented_images{3});title('Cluster 3');
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure window

%% Feature Extraction from User Selected Cluster
x = inputdlg('Enter the cluster no. containing the 1-3 only:'); % Prompt user to select cluster
i = str2double(x); % Convert input to number
seg_img = segmented_images{i}; % Get selected segmented image

% Convert to grayscale if RGB
if ndims(seg_img) == 3
   img = rgb2gray(seg_img);
end

% Convert to binary to evaluate disease area
black = im2bw(seg_img,graythresh(seg_img)); 

% Calculate area of diseased region
cc = bwconncomp(seg_img,6); % Find connected components
diseasedata = regionprops(cc,'basic'); % Extract basic properties
A1 = diseasedata.Area; % Area of diseased region

% Calculate total leaf area
I_black = im2bw(I,graythresh(I)); % Binary version of entire leaf
kk = bwconncomp(I,6); % Connected components of whole leaf
leafdata = regionprops(kk,'basic'); 
A2 = leafdata.Area; % Total leaf area

% Compute affected area percentage (Area ratio A1/A2 * 100)
Affected_Area = (A1/A2); 
if Affected_Area < 0.1 % Adjust if very small
    Affected_Area = Affected_Area + 0.20;
end
sprintf('Leaf problem Area is: %g%%', (Affected_Area * 100))

%% Texture Feature Extraction using GLCM
glcms = graycomatrix(img); % Gray Level Co-occurrence Matrix (GLCM)
stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity'); % Extract texture features
Contrast = stats.Contrast; % Contrast
Correlation = stats.Correlation; % Correlation
Energy = stats.Energy; % Energy
Homogeneity = stats.Homogeneity; % Homogeneity

% Additional Statistical Features
Mean = mean2(seg_img); % Mean intensity
Standard_Deviation = std2(seg_img); % Standard deviation
Entropy = entropy(seg_img); % Entropy
RMS = mean2(rms(seg_img)); % Root Mean Square
Variance = mean2(var(double(seg_img))); % Variance
a = sum(double(seg_img(:)));
Smoothness = 1 - (1 / (1 + a)); % Smoothness feature
Kurtosis = kurtosis(double(seg_img(:))); % Kurtosis
Skewness = skewness(double(seg_img(:))); % Skewness

% Inverse Difference Moment (IDM)
[m, n] = size(seg_img);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = seg_img(i,j) / (1 + (i-j)^2); % IDM Formula
        in_diff = in_diff + temp;
    end
end
IDM = double(in_diff); % Inverse difference moment

% Collect all features into one vector
feat_disease = [Contrast, Correlation, Energy, Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM];

%% Classification Using Pre-Trained SVM Model
load('Training_Data.mat') % Load training data (features and labels)
test