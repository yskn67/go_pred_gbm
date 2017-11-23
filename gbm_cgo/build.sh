#! /bin/bash

go build
install_name_tool -change lib/libxgboost.dylib ../xgboost/lib/libxgboost.dylib gbm_cgo
