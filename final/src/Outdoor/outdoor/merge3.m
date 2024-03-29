% 8500
clear all
% close all


list_all=dir('./8500_testpatch/*png');
image_index=1;

folder_name='./8500_test2/';
if not(exist(folder_name,'dir'))
        mkdir(folder_name)
end


for all_index=1:5
image1=zeros(3072,3072,3,5);

for i=1:7
    img1=imread(['./8500_testpatch/',num2str(image_index-1),'.png']);
    img1=imresize(img1,[1536,3072]);
    image_index=image_index+1;
    start=(i-1)*256+1;
    end1=(i-1)*256+1536;
%     image1(1:1024,start:end1,1:3,i)=img1;
    image1(start:end1,1:3072,1:3,i)=img1;    
end

zz2=sum(image1,4);

zz3=zeros(size(zz2));
start2=1;

ratio=[1,2,3,4,5,6,6,5,4,3,2,1];


zz3=zeros(3072,1024,3,12);

folder=cell(1,12);
for index=1:12
    start2=256*(index-1)+1;
    zz4=zz2(start2:start2+255,1:3072,1:3)/ratio(index);
    zz3(start2:start2+255,1:3072,1:3,index)=zz4;
end

zz4=sum(zz3,4);



size_all=[2833,4657;2833,4657;3052,4706;3122,4576;3122,4776];
% size_all=[2833,4657;2833,4657;2833,4657;2833,4657;2833,4657];
size_all=[3533,3574;3533,3102;2679,3670;2842,3916;3362,3506];


    img1=zz4/255;
    img1=imresize(img1,size_all(all_index,:));

    imwrite(img1, ['./8500_test2/',num2str(all_index+25),'.png']);


end

% 9000
clear all
% close all


list_all=dir('./9000_testpatch/*png');
image_index=1;

folder_name='./9000_test2/';
if not(exist(folder_name,'dir'))
        mkdir(folder_name)
end


for all_index=1:5
image1=zeros(3072,3072,3,5);

for i=1:7
    img1=imread(['./9000_testpatch/',num2str(image_index-1),'.png']);
    img1=imresize(img1,[1536,3072]);
    image_index=image_index+1;
    start=(i-1)*256+1;
    end1=(i-1)*256+1536;
%     image1(1:1024,start:end1,1:3,i)=img1;
    image1(start:end1,1:3072,1:3,i)=img1;    
end

zz2=sum(image1,4);

zz3=zeros(size(zz2));
start2=1;

ratio=[1,2,3,4,5,6,6,5,4,3,2,1];


zz3=zeros(3072,1024,3,12);

folder=cell(1,12);
for index=1:12
    start2=256*(index-1)+1;
    zz4=zz2(start2:start2+255,1:3072,1:3)/ratio(index);
    zz3(start2:start2+255,1:3072,1:3,index)=zz4;
end

zz4=sum(zz3,4);



size_all=[2833,4657;2833,4657;3052,4706;3122,4576;3122,4776];
% size_all=[2833,4657;2833,4657;2833,4657;2833,4657;2833,4657];
size_all=[3533,3574;3533,3102;2679,3670;2842,3916;3362,3506];


    img1=zz4/255;
    img1=imresize(img1,size_all(all_index,:));

    imwrite(img1, ['./9000_test2/',num2str(all_index+25),'.png']);


end

% 9500
clear all
% close all


list_all=dir('./9500_testpatch/*png');
image_index=1;

folder_name='./9500_test2/';
if not(exist(folder_name,'dir'))
        mkdir(folder_name)
end


for all_index=1:5
image1=zeros(3072,3072,3,5);

for i=1:7
    img1=imread(['./9500_testpatch/',num2str(image_index-1),'.png']);
    img1=imresize(img1,[1536,3072]);
    image_index=image_index+1;
    start=(i-1)*256+1;
    end1=(i-1)*256+1536;
%     image1(1:1024,start:end1,1:3,i)=img1;
    image1(start:end1,1:3072,1:3,i)=img1;    
end

zz2=sum(image1,4);

zz3=zeros(size(zz2));
start2=1;

ratio=[1,2,3,4,5,6,6,5,4,3,2,1];


zz3=zeros(3072,1024,3,12);

folder=cell(1,12);
for index=1:12
    start2=256*(index-1)+1;
    zz4=zz2(start2:start2+255,1:3072,1:3)/ratio(index);
    zz3(start2:start2+255,1:3072,1:3,index)=zz4;
end

zz4=sum(zz3,4);



size_all=[2833,4657;2833,4657;3052,4706;3122,4576;3122,4776];
% size_all=[2833,4657;2833,4657;2833,4657;2833,4657;2833,4657];
size_all=[3533,3574;3533,3102;2679,3670;2842,3916;3362,3506];


    img1=zz4/255;
    img1=imresize(img1,size_all(all_index,:));

    imwrite(img1, ['./9500_test2/',num2str(all_index+25),'.png']);


end
