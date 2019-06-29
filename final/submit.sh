mkdir output
cp ./src/Indoor/indoor/indoor_submit/*.jpg output/
cp ./src/Outdoor/outdoor/outdoor_submit/*.jpg output/

cd output/
tar zcvf ../output.tgz *.jpg
