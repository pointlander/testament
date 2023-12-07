// Copyright 2023 The Testament Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"flag"
	"fmt"
	"hash/fnv"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"sync/atomic"

	"github.com/fatih/color"
	. "github.com/pointlander/matrix"
)

const (
	// Batch is the batch size
	Batch = 1
	// Samples is the number of samples per batch
	Samples = 256 / Batch
	// Size is the size of the embedding
	Size = 32
)

// Random is a random variable
type Random struct {
	Mean   float32
	StdDev float32
}

// Set is a set of statistics
type Set [][]Random

// NewStatistics generates a new statistics model
func NewStatistics(inputs, outputs int) Set {
	statistics := make(Set, outputs)
	for i := range statistics {
		for j := 0; j < inputs; j++ {
			statistics[i] = append(statistics[i], Random{
				Mean:   0,
				StdDev: 1,
			})
		}
	}
	return statistics
}

// Sample samples from the statistics
func (s Set) Sample(rng *rand.Rand, inputs, outputs int) []Matrix {
	neurons := make([]Matrix, outputs)
	for j := range neurons {
		neurons[j] = NewMatrix(0, inputs, 1)
		for k := 0; k < inputs; k++ {
			v := float32(rng.NormFloat64())*s[j][k].StdDev + s[j][k].Mean
			if v > 0 {
				v = 1
			} else {
				v = -1
			}
			neurons[j].Data = append(neurons[j].Data, v)
		}
	}
	return neurons
}

// Net is a net
type Net struct {
	window  int64
	Inputs  int
	Outputs int
	Rng     *rand.Rand
	Q       Set
	K       Set
	V       Set
}

// NewNet makes a new network
func NewNet(seed int64, window int64, inputs, outputs int) Net {
	rng := rand.New(rand.NewSource(seed))
	return Net{
		window:  window,
		Inputs:  inputs,
		Outputs: outputs,
		Rng:     rng,
		Q:       NewStatistics(inputs, outputs),
		K:       NewStatistics(inputs, outputs),
		V:       NewStatistics(inputs, outputs),
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
	Out     Matrix
}

// CalculateStatistics calculates the statistics of systems
func (n Net) CalculateStatistics(systems []Sample) Set {
	window := atomic.LoadInt64(&n.window)
	statistics := make(Set, n.Outputs)
	for i := range statistics {
		for j := 0; j < n.Inputs; j++ {
			statistics[i] = append(statistics[i], Random{
				Mean:   0,
				StdDev: 0,
			})
		}
	}
	for i := range systems[:window] {
		for j := range systems[i].Neurons {
			for k, value := range systems[i].Neurons[j].Data {
				statistics[j][k].Mean += value
			}
		}
	}
	for i := range statistics {
		for j := range statistics[i] {
			statistics[i][j].Mean /= float32(window)
		}
	}
	for i := range systems[:window] {
		for j := range systems[i].Neurons {
			for k, value := range systems[i].Neurons[j].Data {
				diff := statistics[j][k].Mean - value
				statistics[j][k].StdDev += diff * diff
			}
		}
	}
	for i := range statistics {
		for j := range statistics[i] {
			statistics[i][j].StdDev /= float32(window)
			statistics[i][j].StdDev = float32(math.Sqrt(float64(statistics[i][j].StdDev)))
		}
	}
	return statistics
}

// Fire runs the network
func (n *Net) Fire(input Matrix) Matrix {
	q := NewMatrix(0, n.Outputs, Samples)
	k := NewMatrix(0, n.Outputs, Samples)
	v := NewMatrix(0, n.Outputs, Samples)
	systemsQ := make([]Sample, 0, 8)
	systemsK := make([]Sample, 0, 8)
	systemsV := make([]Sample, 0, 8)
	for i := 0; i < Samples; i++ {
		neurons := n.Q.Sample(n.Rng, n.Inputs, n.Outputs)
		outputs := NewMatrix(0, n.Outputs, 1)
		for j := range neurons {
			out := MulT(neurons[j], input)
			q.Data = append(q.Data, out.Data[0])
			outputs.Data = append(outputs.Data, out.Data[0])
		}
		systemsQ = append(systemsQ, Sample{
			Neurons: neurons,
			Outputs: outputs,
		})
	}
	for i := 0; i < Samples; i++ {
		neurons := n.K.Sample(n.Rng, n.Inputs, n.Outputs)
		outputs := NewMatrix(0, n.Outputs, 1)
		for j := range neurons {
			out := MulT(neurons[j], input)
			k.Data = append(k.Data, out.Data[0])
			outputs.Data = append(outputs.Data, out.Data[0])
		}
		systemsK = append(systemsK, Sample{
			Neurons: neurons,
			Outputs: outputs,
		})
	}
	for i := 0; i < Samples; i++ {
		neurons := n.V.Sample(n.Rng, n.Inputs, n.Outputs)
		outputs := NewMatrix(0, n.Outputs, 1)
		for j := range neurons {
			out := MulT(neurons[j], input)
			v.Data = append(v.Data, out.Data[0])
			outputs.Data = append(outputs.Data, out.Data[0])
		}
		systemsV = append(systemsV, Sample{
			Neurons: neurons,
			Outputs: outputs,
		})
	}
	entropies := SelfEntropy(q, k, v)
	for i, entropy := range entropies {
		systemsQ[i].Entropy = entropy
		systemsK[i].Entropy = entropy
		systemsV[i].Entropy = entropy
	}
	sort.Slice(systemsQ, func(i, j int) bool {
		return systemsQ[i].Entropy < systemsQ[j].Entropy
	})
	sort.Slice(systemsK, func(i, j int) bool {
		return systemsK[i].Entropy < systemsK[j].Entropy
	})
	sort.Slice(systemsV, func(i, j int) bool {
		return systemsV[i].Entropy < systemsV[j].Entropy
	})

	n.Q = n.CalculateStatistics(systemsQ)
	n.K = n.CalculateStatistics(systemsK)
	n.V = n.CalculateStatistics(systemsV)
	return systemsV[0].Outputs
}

var (
	// FlagFile is the file to process
	FlagFile = flag.String("f", "10.txt.utf-8.bz2", "the file to process")
	// FlagWander is wandering mode
	FlagWander = flag.Bool("w", false, "wander mode")
)

func main() {
	flag.Parse()

	color.Blue("Hello World!")

	data := []byte{}
	if strings.HasSuffix(*FlagFile, ".bz2") {
		input, err := os.Open(*FlagFile)
		if err != nil {
			panic(err)
		}
		defer input.Close()
		reader := bzip2.NewReader(input)
		d, err := ioutil.ReadAll(reader)
		if err != nil {
			panic(err)
		}
		fmt.Println(len(d))
		runes := []rune(string(d))
		count := 0
		for _, v := range runes {
			if v < 256 {
				data = append(data, byte(v))
			} else {
				count++
			}
		}
		fmt.Println("unicode", count)
	} else {
		input, err := os.Open(*FlagFile)
		if err != nil {
			panic(err)
		}
		defer input.Close()
		d, err := ioutil.ReadAll(input)
		if err != nil {
			panic(err)
		}
		data = d
	}

	if *FlagWander {
		net := NewNet(2, 8, Size, 16)
		in := NewMatrix(0, Size, Batch)
		in.Data = in.Data[:cap(in.Data)]
		position, length := 0, len(data)
		seen := make(map[int]bool, 8)
		h := fnv.New32()
		for len(seen) != length {
			for i := 0; i < Batch; i++ {
				h.Reset()
				h.Write(data[position+i : position+i+1])
				rng := rand.New(rand.NewSource(int64(h.Sum32())))
				embedding := [256]float32{}
				sum := 0.0
				for i := range embedding {
					v := rng.NormFloat64()
					sum += v * v
					embedding[i] = float32(v)
				}
				length := float32(math.Sqrt(sum))
				for i, v := range embedding {
					embedding[i] = v / length
				}
				copy(in.Data[i*Size:(i+1)*Size], embedding[:])
			}
			out := net.Fire(in)
			c := 0
			for i, v := range out.Data {
				if v > 0 {
					c |= 1 << i
				}
			}
			seen[position] = true
			if len(seen) == length {
				break
			}
			position = c % length
			if seen[position] {
				break
			}
			for seen[position] {
				position = (position + 1) % length
			}
			fmt.Println(position, string(data[position]))
		}
		return
	}

	test := func(iterations int) {
		net := NewNet(2, 8, Size, 3)
		in := NewMatrix(0, Size, Batch)
		in.Data = in.Data[:cap(in.Data)]
		position := 0
		h := fnv.New32()
		for position < iterations {
			for i := 0; i < Batch; i++ {
				h.Reset()
				h.Write(data[position+i : position+i+1])
				rng := rand.New(rand.NewSource(int64(h.Sum32())))
				embedding := [256]float32{}
				sum := 0.0
				for i := range embedding {
					v := rng.NormFloat64()
					sum += v * v
					embedding[i] = float32(v)
				}
				length := float32(math.Sqrt(sum))
				for i, v := range embedding {
					embedding[i] = v / length
				}
				copy(in.Data[i*Size:(i+1)*Size], embedding[:])
			}
			out := net.Fire(in)
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
			symbol := ""
			switch c {
			case 0:
				symbol = color.BlackString(string(data[position]))
			case 1:
				symbol = color.BlueString(string(data[position]))
			case 2:
				symbol = color.RedString(string(data[position]))
			case 3:
				symbol = color.GreenString(string(data[position]))
			case 4:
				symbol = color.CyanString(string(data[position]))
			case 5:
				symbol = color.YellowString(string(data[position]))
			case 6:
				symbol = color.MagentaString(string(data[position]))
			case 7:
				symbol = color.HiMagentaString(string(data[position]))
			}
			fmt.Printf(symbol)
			position++
		}
	}

	test(len(data))
}
