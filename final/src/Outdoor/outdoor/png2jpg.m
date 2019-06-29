% 8500
liste = dir('./8500_submitted/*.png');

folder_name='./out_8500/';

if not(exist(folder_name,'dir'))
        mkdir(folder_name)
end

for i=1:size(liste, 1)
    imgName = strcat('./8500_submitted/', liste(i).name);
    C = imread(imgName);
    imwrite(C, strcat('./out_8500/', int2str(i+5), '.jpg'), 'jpg');
end

% 9000
liste = dir('./9000_submitted/*.png');

folder_name='./out_9000/';

if not(exist(folder_name,'dir'))
        mkdir(folder_name)
end

for i=1:size(liste, 1)
    imgName = strcat('./9000_submitted/', liste(i).name);
    C = imread(imgName);
    imwrite(C, strcat('./out_9000/', int2str(i+5), '.jpg'), 'jpg');
end

% 9500
liste = dir('./9500_submitted/*.png');

folder_name='./out_9500/';

if not(exist(folder_name,'dir'))
        mkdir(folder_name)
end

for i=1:size(liste, 1)
    imgName = strcat('./9500_submitted/', liste(i).name);
    C = imread(imgName);
    imwrite(C, strcat('./out_9500/', int2str(i+5), '.jpg'), 'jpg');
end