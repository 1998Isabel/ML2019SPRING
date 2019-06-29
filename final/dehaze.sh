python3 ./src/Indoor/dehazetest.py --dataroot ./src/Indoor/sample/indoortest2 --valDataroo ./src/Indoor/sample/indoortest2 --netG ./src/Indoor/pretrained-model/netG_epoch_23000.pth
python3 ./src/Indoor/dehazetest1.py --dataroot ./src/Indoor/sample/indoortest3 --valDataroo ./src/Indoor/sample/indoortest3 --netG ./src/Indoor/pretrained-model/netG_epoch_23000.pth

python3 ./src/Outdoor/dehazetest.py --dataroot ./src/Outdoor/sample/outdoortest2 --valDataroo ./src/Outdoor/sample/outdoortest2 --netG ./src/Outdoor/pretrained-model/netG_epoch_8500.pth --directory ./src/Outdoor/outdoor/8500_test
python3 ./src/Outdoor/dehazetest1.py --dataroot ./src/Outdoor/sample/outdoortest2pat --valDataroo ./src/Outdoor/sample/outdoortest2pat --netG ./src/Outdoor/pretrained-model/netG_epoch_8500.pth --directory ./src/Outdoor/outdoor/8500_testpatch

python3 ./src/Outdoor/dehazetest.py --dataroot ./src/Outdoor/sample/outdoortest2 --valDataroo ./src/Outdoor/sample/outdoortest2 --netG ./src/Outdoor/pretrained-model/netG_epoch_9000.pth --directory ./src/Outdoor/outdoor/9000_test
python3 ./src/Outdoor/dehazetest1.py --dataroot ./src/Outdoor/sample/outdoortest2pat --valDataroo ./src/Outdoor/sample/outdoortest2pat --netG ./src/Outdoor/pretrained-model/netG_epoch_9000.pth --directory ./src/Outdoor/outdoor/9000_testpatch

python3 ./src/Outdoor/dehazetest.py --dataroot ./src/Outdoor/sample/outdoortest2 --valDataroo ./src/Outdoor/sample/outdoortest2 --netG ./src/Outdoor/pretrained-model/netG_epoch_9500.pth --directory ./src/Outdoor/outdoor/9500_test
python3 ./src/Outdoor/dehazetest1.py --dataroot ./src/Outdoor/sample/outdoortest2pat --valDataroo ./src/Outdoor/sample/outdoortest2pat --netG ./src/Outdoor/pretrained-model/netG_epoch_9500.pth --directory ./src/Outdoor/outdoor/9500_testpatch
