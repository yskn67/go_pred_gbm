package gbm

import (
	"bufio"
	"encoding/csv"
	"io"
	"os"
	"strconv"
	"strings"
	"testing"
)

func TestNewModel(t *testing.T) {
	m, err := NewModel("../../model/dump_model.json")
	if m == nil {
		t.Error("ERR: model is not loaded")
	}
	if err != nil {
		t.Error("ERR: get error when model is loaded")
	}
}

func TestPredict(t *testing.T) {
	m, err := NewModel("../../model/dump_model.json")

	file, err := os.Open("../../data/test.txt")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()

	r := bufio.NewReaderSize(file, 4096)
	for line := ""; err == nil; line, err = r.ReadString('\n') {
		if len(line) <= 0 {
			continue
		}
		line = strings.TrimRight(line, "\n")
		rcsv := csv.NewReader(strings.NewReader(line))
		rcsv.Comma = '\t'

		record, _ := rcsv.ReadAll()
		d := make([]float64, len(record[0])-1)
		for i := 1; i < len(record[0]); i++ {
			d[i-1], _ = strconv.ParseFloat(record[0][i], 64)
		}

		p := m.Predict(d)
		if p < 0 {
			t.Error("ERR: predicted values is under 0")
		} else if p > 1 {
			t.Error("ERR: predicted values is over 1")
		}
	}
	if err != io.EOF {
		t.Fatal(err)
	}
}

func BenchmarkPredict(b *testing.B) {
	m, _ := NewModel("../../model/dump_model.json")
	file, _ := os.Open("../../data/test_oneline.txt")
	defer file.Close()

	d := make([]float64, 100)
	r := bufio.NewReaderSize(file, 4096)
	var err error
	for line := ""; err == nil; line, err = r.ReadString('\n') {
		if len(line) <= 0 {
			continue
		}
		line = strings.TrimRight(line, "\n")
		rcsv := csv.NewReader(strings.NewReader(line))
		rcsv.Comma = '\t'

		record, _ := rcsv.ReadAll()
		for i := 1; i < len(record[0]); i++ {
			d[i-1], _ = strconv.ParseFloat(record[0][i], 64)
		}
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = m.Predict(d)
	}
}
