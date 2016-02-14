#! /bin/bash

wget \
--recursive \
--page-requisites \
--html-extension \
--no-parent \
--reject *.zip \
--reject *.html \
--reject *.txt \
http://www.cs.toronto.edu/~vmnih/data/

mv www.cs.toronto.edu/~vmnih/data ./
rm -rf www.cs.toronto.edu

mv data/mass_merged/test/map data/mass_merged/test/map_orig
cd data/mass_merged/test; wget https://www.dropbox.com/s/yk6d4garyz3nm19/multi_test_map.tar.gz?dl=0;
tar zxvf multi_test_map.tar.gz; rm -rf multi_test_map.tar.gz
