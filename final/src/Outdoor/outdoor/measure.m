% 8500
clear all
close all

list_all=dir('./8500_test/*png');
list_all2=dir('./8500_test2/*png');

size_all=[2942,2426;3412,4056;3352,4846;3482,2636;2602,3696];


folder_name='./8500_submitted/';
if not(exist(folder_name,'dir'))
        mkdir(folder_name)
end

for i=1:5
    img1=imread(['./8500_test/',list_all(i).name]);

    hazy2=imread(['./8500_test2/',list_all2(i).name]);
    
    img1=imresize(img1,size_all(i,:));
    hazy2=imresize(hazy2,size_all(i,:));

    img3=img1/2+hazy2/2;

    imwrite(img3, ['./8500_submitted/',num2str(i+40),'.png']);
end

%9000
clear all
close all

list_all=dir('./9000_test/*png');
list_all2=dir('./9000_test2/*png');

size_all=[2942,2426;3412,4056;3352,4846;3482,2636;2602,3696];


folder_name='./9000_submitted/';
if not(exist(folder_name,'dir'))
        mkdir(folder_name)
end

for i=1:5
    img1=imread(['./9000_test/',list_all(i).name]);

    hazy2=imread(['./9000_test2/',list_all2(i).name]);
    
    img1=imresize(img1,size_all(i,:));
    hazy2=imresize(hazy2,size_all(i,:));

    img3=img1/2+hazy2/2;

    imwrite(img3, ['./9000_submitted/',num2str(i+40),'.png']);
end

%9500
clear all
close all

list_all=dir('./9500_test/*png');
list_all2=dir('./9500_test2/*png');

size_all=[2942,2426;3412,4056;3352,4846;3482,2636;2602,3696];


folder_name='./9500_submitted/';
if not(exist(folder_name,'dir'))
        mkdir(folder_name)
end

for i=1:5
    img1=imread(['./9500_test/',list_all(i).name]);

    hazy2=imread(['./9500_test2/',list_all2(i).name]);
    
    img1=imresize(img1,size_all(i,:));
    hazy2=imresize(hazy2,size_all(i,:));

    img3=img1/2+hazy2/2;

    imwrite(img3, ['./9500_submitted/',num2str(i+40),'.png']);
end
