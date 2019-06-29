clear all
close all
list_all=dir('./out_8500/*jpg');

folder_name='./outdoor_submit/';

if not(exist(folder_name,'dir'))
        mkdir(folder_name)
end

for i=1:5
    hazy=imread(['./out_8500/',list_all(i).name]);
    hazy2=imread(['./out_9000/',list_all(i).name]);
    hazy3=imread(['./out_9500/',list_all(i).name]);


    img1= hazy/3 + hazy2/3 + hazy3/3;
    name = list_all(i).name;
    imwrite([img1],['./outdoor_submit/',name]);

    
end