liste = dir('./our_cvprw_submitted/*.png');

folder_name='./indoor_submit/';

if not(exist(folder_name,'dir'))
        mkdir(folder_name)
end

for i=1:size(liste, 1)
    imgName = strcat('./our_cvprw_submitted/', liste(i).name);
    C = imread(imgName);
    imwrite(C, strcat('./indoor_submit/', int2str(i), '.jpg'), 'jpg');
end