#! /bin/sh

cd `dirname $0`
git clone --recursive https://github.com/dmlc/xgboost.git  
cd xgboost  
./build.sh
pip install -e python-package 
cd python-package
python setup.py install
