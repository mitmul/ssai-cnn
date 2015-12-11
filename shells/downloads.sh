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
