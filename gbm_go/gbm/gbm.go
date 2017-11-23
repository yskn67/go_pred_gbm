package gbm

import (
	"bytes"
	"encoding/json"
	"os"
)

const BUFSIZE = 1024

type GbLeaf struct {
	Split          *int     `json:"split"`
	SplitCondition *float64 `json:"split_condition"`
	Yes            *int     `json:"yes"`
	No             *int     `json:"no"`
	Missing        *int     `json:"missing"`
	Leaf           *float64 `json:"leaf"`
}

type GbTree []GbLeaf

func (gbt GbTree) Len() int {
	return len(gbt)
}

type Model []GbTree

func (m Model) Len() int {
	return len(m)
}
func (m Model) Predict(d []float64) float64 {
	n_tree := m.Len()
	sum_ret := float64(0)
	for i := 0; i < n_tree; i++ {
		sum_ret += predict(m[i], d, 0)
	}
	return sum_ret + 0.5
}

func predict(gbt GbTree, d []float64, leaf int) float64 {
	if gbt[leaf].Leaf != nil {
		return *(gbt[leaf].Leaf)
	}

	if d[*(gbt[leaf].Split)] < *(gbt[leaf].SplitCondition) {
		return predict(gbt, d, *(gbt[leaf].Yes))
	} else {
		return predict(gbt, d, *(gbt[leaf].No))
	}
}

func NewModel(path string) (*Model, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var jbuf bytes.Buffer
	buf := make([]byte, BUFSIZE)
	for {
		n, err := file.Read(buf)
		if n == 0 {
			break
		}
		if err != nil {
			return nil, err
		}

		jbuf.Write(buf[:n])
	}

	model := new(Model)
	if err := json.Unmarshal(jbuf.Bytes(), model); err != nil {
		return nil, err
	}

	return model, nil
}
