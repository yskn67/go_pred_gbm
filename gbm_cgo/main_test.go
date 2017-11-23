package main

import "testing"

func BenchmarkPredict(b *testing.B) {
	bst, err := NewBooster()
	if err != nil {
		b.Fatal(err)
	}
	defer bst.Free()

	err = bst.LoadModel("../model/dump.model")
	if err != nil {
		b.Fatal(err)
	}

	dm, err := NewDMatrix("../data/test_libsvm_oneline.txt")
	if err != nil {
		b.Fatal(err)
	}
	defer dm.Free()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = bst.Predict(dm)
	}
}
