wget -O all.tar.gz https://www.dropbox.com/s/ywexg0718k8jrab/all.tar.gz?dl=1
tar zxf all.tar.gz

tar zxf Testing_Images.tar.gz
tar zxf indoortest2.tar.gz --directory ./src/Indoor/sample
tar zxf indoortest3.tar.gz --directory ./src/Indoor/sample
tar zxf outdoortest2.tar.gz --directory ./src/Outdoor/sample
tar zxf outdoortest2pat.tar.gz --directory ./src/Outdoor/sample
mv netG_epoch_23000.pth ./src/Indoor/pretrained-model/
mv netG_epoch_8500.pth netG_epoch_9000.pth netG_epoch_9500.pth ./src/Outdoor/pretrained-model/

rm *.tar.gz
