package main

import (
	"./gbm"
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

func main() {
	m, err := gbm.NewModel("../model/dump_model.json")

	file, err := os.Open("../data/test.txt")
	if err != nil {
		panic(err)
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
		fmt.Println(p)
	}
	if err != io.EOF {
		panic(err)
	}
}
