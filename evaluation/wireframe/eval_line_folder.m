clear;
clc;
close all;


dSetPath = 'line_mat/';
imagesPath = 'images/';
resultPath = fullfile('../../result/wireframe_0.5_0.5/');

outFile = fullfile('1_0.5_0.5.mat');

maxDist = 0.01;
divide_eps = 1e-15;
%lineThresh = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130,...
%    140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 255];
lineThresh = [2, 6, 10, 20, 30, 50, 80, 100, 150, 200, 250, 255];

nLineThresh = size(lineThresh, 2);
sumprecisions = zeros(nLineThresh, 1);
sumrecalls = zeros(nLineThresh, 1);
nsamples = zeros(nLineThresh, 1);

listing = dir(imagesPath);
numResults = size(listing, 1);

%for index=1:numResults
for index=1:numResults
    filename = listing(index).name;
    if length(filename) == 1 || length(filename) == 2
        continue;
    end
    filename = filename(1:end-4);
    disp([num2str(index), ' == ', filename])
    gtname = [dSetPath, filename, '_line.mat'];
    imgname = [imagesPath, filename, '.jpg'];
    
    I = imread(imgname);
    height = size(I,1);
    width = size(I,2);
    
    %% convert GT lines to binary map
    gtlines = load(gtname);
    gtlines = gtlines.lines;

    ne = size(gtlines,1);
    edgemap0 = zeros(height, width);
    for k = 1:ne
        x1 = gtlines(k,1);
        x2 = gtlines(k,3);
        y1 = gtlines(k,2);
        y2 = gtlines(k,4);

        vn = ceil(sqrt((x1-x2)^2+(y1-y2)^2));
        cur_edge = [linspace(y1,y2,vn).', linspace(x1,x2,vn).'];
        for j = 1:size(cur_edge,1)
            yy = round(cur_edge(j,1));
            xx = round(cur_edge(j,2));
            if yy <= 0
                yy = 1;
            end
            if xx <= 0
                xx = 1;
            end
            edgemap0(yy,xx) = 1;
        end
    end

    parfor m=1:nLineThresh
    %for m=1:nLineThresh
        resultname = [resultPath, '/', num2str(lineThresh(m)), '/', filename, '.mat'];
        resultlines = load(resultname);
        resultlines = resultlines.lines;
        ne = size(resultlines,1);
        disp([' ', num2str(lineThresh(m)), ' #lines: ' num2str(ne)]);

        edgemap1 = zeros(height, width);
        for k = 1:ne
            x1 = resultlines(k,1);
            y1 = resultlines(k,2);
            x2 = resultlines(k,3);                    
            y2 = resultlines(k,4);

            vn = ceil(sqrt((x1-x2)^2+(y1-y2)^2));
            cur_edge = [linspace(y1,y2,vn).', linspace(x1,x2,vn).'];
            for j = 1:size(cur_edge,1)
                yy = round(cur_edge(j,1));
                xx = round(cur_edge(j,2));
                if yy <= 0
                    yy = 1;
                end
                if xx <= 0
                    xx = 1;
                end
                if yy > height
                    yy = height;                    
                end
                if xx > width
                    xx = width;
                end
                edgemap1(yy,xx) = 1;
            end
        end

        [matchE1,matchG1] = correspondPixels(edgemap1,edgemap0,maxDist);
        matchE = double(matchE1>0);
        matchG = double(matchG1>0);

        % compute recall (summed over each gt image)
        cntR = sum(matchG(:)); sumR = sum(edgemap0(:)); 
        recall = cntR / (sumR + divide_eps);
        % compute precision (edges can match any gt image)
        cntP = sum(matchE(:)); sumP = sum(edgemap1(:)); precision = cntP / (sumP + divide_eps);
        disp(['===== filename = ' filename ', precision = ' num2str(precision) ', recall = ' num2str(recall)]);
        sumprecisions(m, 1) = sumprecisions(m, 1) + precision;
        sumrecalls(m, 1) = sumrecalls(m, 1) + recall;
        nsamples(m, 1) = nsamples(m, 1) + 1;
    end
end
disp([sumprecisions; sumrecalls])
save(outFile, 'sumprecisions', 'sumrecalls', 'nsamples');



