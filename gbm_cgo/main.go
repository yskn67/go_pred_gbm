package main

/*
#cgo CFLAGS: -I${SRCDIR}/../xgboost/dmlc-core/include -I${SRCDIR}/../xgboost/rabit/include -I${SRCDIR}/../xgboost/include
#cgo LDFLAGS: -pthread -lm -L${SRCDIR}/../xgboost/lib -lxgboost
#include <stdlib.h>
#include <xgboost/c_api.h>
*/
import "C"
import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"unsafe"
)

const mpath = "../model/dump.model"
const dpath = "../data/test_libsvm.txt"

func Conv2DCArray(mtx [][]float32) []*C.float {
	carray := make([]*C.float, len(mtx))
	for i := range carray {
		carray[i] = (*C.float)(&mtx[i][0])
	}
	return carray
}

type Booster struct {
	Handle C.BoosterHandle
}

func NewBooster() (*Booster, error) {
	var handle C.BoosterHandle
	dmatrix := [1]C.DMatrixHandle{}
	ret := C.XGBoosterCreate((*C.DMatrixHandle)(&dmatrix[0]), C.bst_ulong(0), &handle)
	if ret != 0 {
		return nil, errors.New("Error: Could not create new booster")
	}
	bst := Booster{handle}
	return &bst, nil
}

func (bst *Booster) LoadModel(path string) error {
	if bst.Handle == nil {
		return errors.New("Error: booster is deleted")
	}
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	ret := C.XGBoosterLoadModel(bst.Handle, cpath)
	if ret != 0 {
		return errors.New("Error: Could not load booster")
	}
	return nil
}

func (bst *Booster) Predict(dm *DMatrix) ([]float32, error) {
	nrow, err := dm.NumRow()
	if err != nil {
		return nil, err
	}

	cpreds := make([]*C.float, nrow)
	for i := range cpreds {
		cpred := C.float(0)
		cpreds[i] = &cpred
	}
	length := C.bst_ulong(0)
	ret := C.XGBoosterPredict(bst.Handle, dm.Handle, C.int(0), C.uint(0), &length, (**C.float)(&cpreds[0]))
	if ret != 0 {
		return nil, errors.New("Error: Could not predict")
	}

	preds := make([]float32, nrow)
	b := C.GoBytes(unsafe.Pointer(cpreds[0]), C.int(C.sizeof_float*nrow))
	err = binary.Read(bytes.NewReader(b), binary.LittleEndian, preds)
	if err != nil {
		return nil, errors.New("Error: Could not convert predicted values")
	}

	return preds, nil
}

func (bst *Booster) Free() error {
	if bst.Handle != nil {
		ret := C.XGBoosterFree(bst.Handle)
		if ret != 0 {
			return errors.New("Error: Could not delete booster")
		}
		bst.Handle = nil
	}
	return nil
}

type DMatrix struct {
	Handle C.DMatrixHandle
}

func NewDMatrix(path string) (*DMatrix, error) {
	var handle C.DMatrixHandle
	cpath := C.CString(path)
	C.free(unsafe.Pointer(cpath))

	ret := C.XGDMatrixCreateFromFile(cpath, C.int(1), &handle)
	if ret != 0 {
		return nil, errors.New("Error: Could not create new dmatrix")
	}

	return &DMatrix{handle}, nil
}

func (dm *DMatrix) NumRow() (int, error) {
	nrow := C.bst_ulong(0)
	ret := C.XGDMatrixNumRow(dm.Handle, &nrow)
	if ret != 0 {
		return 0, errors.New("Error: Could not get dmatrix row number")
	}
	return (int)(nrow), nil
}

func (dm *DMatrix) NumCol() (int, error) {
	ncol := C.bst_ulong(0)
	ret := C.XGDMatrixNumCol(dm.Handle, &ncol)
	if ret != 0 {
		return 0, errors.New("Error: Could not get dmatrix col number")
	}
	return (int)(ncol), nil
}

func (dm *DMatrix) Free() error {
	if dm.Handle != nil {
		C.free(unsafe.Pointer(dm.Handle))
		// TODO: fix type
		/*
			ret := C.XGDMatrixFree(dm.Handle)
			if ret != 0 {
				return errors.New("Error: Could not delete dmatrix")
			}
		*/
		dm.Handle = nil
	}
	return nil
}

func main() {
	bst, err := NewBooster()
	if err != nil {
		panic(err)
	}
	defer bst.Free()
	err = bst.LoadModel(mpath)
	if err != nil {
		panic(err)
	}

	dm, err := NewDMatrix(dpath)
	defer dm.Free()
	if err != nil {
		panic(err)
	}

	preds, err := bst.Predict(dm)
	if err != nil {
		panic(err)
	}

	for i := range preds {
		fmt.Println(preds[i])
	}
}
