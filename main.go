// Copyright 2023 The Testament Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"flag"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync/atomic"

	"github.com/fatih/color"
	. "github.com/pointlander/matrix"
)

const (
	// Batch is the batch size
	Batch = 1
	// Samples is the number of samples per batch
	Samples = 256 / Batch
)

// Random is a random variable
type Random struct {
	Mean   float32
	StdDev float32
}

// Net is a net
type Net struct {
	window       int64
	Inputs       int
	Outputs      int
	Rng          *rand.Rand
	Distribution [][]Random
}

// NewNet makes a new network
func NewNet(seed int64, window int64, inputs, outputs int) Net {
	rng := rand.New(rand.NewSource(seed))
	distribution := make([][]Random, outputs)
	for i := range distribution {
		for j := 0; j < inputs; j++ {
			distribution[i] = append(distribution[i], Random{
				Mean:   0,
				StdDev: 1,
			})
		}
	}
	return Net{
		window:       window,
		Inputs:       inputs,
		Outputs:      outputs,
		Rng:          rng,
		Distribution: distribution,
	}
}

// Set window sets the window
func (n *Net) SetWindow(window int64) {
	atomic.StoreInt64(&n.window, window)
}

// Sample is a sample of a random neural network
type Sample struct {
	Entropy float32
	Neurons []Matrix
	Outputs Matrix
}

// Fire runs the network
func (n *Net) Fire(input Matrix) Matrix {
	rng, distribution, window := n.Rng, n.Distribution, atomic.LoadInt64(&n.window)
	output := NewMatrix(0, n.Outputs, Batch*Samples)

	systems := make([]Sample, 0, 8)
	for i := 0; i < Samples; i++ {
		neurons := make([]Matrix, n.Outputs)
		for j := range neurons {
			neurons[j] = NewMatrix(0, n.Inputs, 1)
			for k := 0; k < n.Inputs; k++ {
				v := float32(rng.NormFloat64())*distribution[j][k].StdDev + distribution[j][k].Mean
				if v > 0 {
					v = 1
				} else {
					v = -1
				}
				neurons[j].Data = append(neurons[j].Data, v)
			}
		}
		outputs := make([]Matrix, Batch)
		for j := range outputs {
			outputs[j] = NewMatrix(0, n.Outputs, 1)
		}
		for j := range neurons {
			out := MulT(neurons[j], input)
			for k, value := range out.Data {
				outputs[k].Data = append(outputs[k].Data, value)
			}
		}
		for j := range outputs {
			output.Data = append(output.Data, outputs[j].Data...)
			systems = append(systems, Sample{
				Neurons: neurons,
				Outputs: outputs[j],
			})
		}
	}
	entropies := SelfEntropy(output, output, output)
	for i, entropy := range entropies {
		systems[i].Entropy = entropy
	}
	sort.Slice(entropies, func(i, j int) bool {
		return systems[i].Entropy < systems[j].Entropy
	})
	next := make([][]Random, n.Outputs)
	for i := range next {
		for j := 0; j < n.Inputs; j++ {
			next[i] = append(next[i], Random{
				Mean:   0,
				StdDev: 0,
			})
		}
	}
	for i := range systems[:window] {
		for j := range systems[i].Neurons {
			for k, value := range systems[i].Neurons[j].Data {
				next[j][k].Mean += value
			}
		}
	}
	for i := range next {
		for j := range next[i] {
			next[i][j].Mean /= float32(window)
		}
	}
	for i := range systems[:window] {
		for j := range systems[i].Neurons {
			for k, value := range systems[i].Neurons[j].Data {
				diff := next[j][k].Mean - value
				next[j][k].StdDev += diff * diff
			}
		}
	}
	for i := range next {
		for j := range next[i] {
			next[i][j].StdDev /= float32(window)
			next[i][j].StdDev = float32(math.Sqrt(float64(next[i][j].StdDev)))
		}
	}
	n.Distribution = next
	outputs := NewMatrix(0, n.Outputs, Batch)
	for i := 0; i < Batch; i++ {
		outputs.Data = append(outputs.Data, systems[i].Outputs.Data...)
	}
	return outputs
}

var (
	// FlagLess less than mode
	FlagLess = flag.Bool("less", false, "less than mode")
	// FlagTest test mode
	FlagTest = flag.Bool("test", false, "test mode")
)

func main() {
	flag.Parse()

	color.Blue("Hello World!")

	input, err := os.Open("10.txt.utf-8.bz2")
	if err != nil {
		panic(err)
	}
	defer input.Close()
	reader := bzip2.NewReader(input)
	data, err := ioutil.ReadAll(reader)
	if err != nil {
		panic(err)
	}
	fmt.Println(len(data))
	runes := []rune(string(data))
	data = []byte{}
	count := 0
	for _, v := range runes {
		if v < 256 {
			data = append(data, byte(v))
		} else {
			count++
		}
	}
	fmt.Println("unicode", count)

	embedding := make([][]float32, 256)
	for i := range embedding {
		embedding[i] = make([]float32, 256)
	}
	for i, v := range data {
		if i > 0 {
			embedding[v][data[i-1]]++
		}
		if i < len(data)-1 {
			embedding[v][data[i+1]]++
		}
	}
	for i := range embedding {
		sum := 0.0
		for _, value := range embedding[i] {
			sum += float64(value) * float64(value)
		}
		length := math.Sqrt(sum)
		if length == 0 {
			continue
		}
		for j := range embedding[i] {
			embedding[i][j] /= float32(length)
		}
	}

	test := func(iterations int) {
		nets := NewNet(1, 8, 256, 16)
		net := NewNet(2, 8, 16, 3)
		in := NewMatrix(0, 256, Batch)
		in.Data = in.Data[:cap(in.Data)]
		position := 0
		//rng := rand.New(rand.NewSource(1))
		for position < iterations {
			for i := 0; i < Batch; i++ {
				copy(in.Data[i*256:(i+1)*256], embedding[data[position+i]])
			}
			out := nets.Fire(in)
			out = net.Fire(out)
			c := 0
			if out.Data[0] > 0 {
				c |= 1
			}
			if out.Data[1] > 0 {
				c |= 2
			}
			if out.Data[2] > 0 {
				c |= 4
			}
			if c > 6 {
				c = 6
			}
			symbol := ""
			switch c {
			case 0:
				symbol = color.BlackString(string(data[position]))
			case 1:
				symbol = color.BlueString(string(data[position]))
			case 2:
				symbol = color.CyanString(string(data[position]))
			case 3:
				symbol = color.GreenString(string(data[position]))
			case 4:
				symbol = color.RedString(string(data[position]))
			case 5:
				symbol = color.YellowString(string(data[position]))
			case 6:
				symbol = color.WhiteString(string(data[position]))
			}
			fmt.Printf(symbol)
			position++
		}
	}

	test(len(data))
}
