// Copyright 2023 The Testament Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"encoding/gob"
	"flag"
	"fmt"
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
	// FlagFile is the file to process
	FlagFile = flag.String("f", "10.txt.utf-8.bz2", "the file to process")
	// FlagLearn is the learning mode
	FlagLearn = flag.Bool("learn", false, "learning mode")
	// FlagWander is wandering mode
	FlagWander = flag.Bool("w", false, "wander mode")
)

func main() {
	flag.Parse()

	color.Blue("Hello World!")

	var embedding [256][256]float32
	if *FlagLearn {
		var process func(file string)
		process = func(file string) {
			fmt.Println(file)
			dirs, err := os.ReadDir(file)
			if err != nil {
				panic(err)
			}
			for _, v := range dirs {
				if v.IsDir() {
					process(file + v.Name() + "/")
				} else {
					if strings.HasSuffix(v.Name(), ".go") {
						fmt.Println(v.Name())
						input, err := os.Open(file + v.Name())
						if err != nil {
							panic(err)
						}
						data, err := ioutil.ReadAll(input)
						if err != nil {
							panic(err)
						}
						for i, v := range data {
							if i > 0 {
								embedding[v][data[i-1]]++
							}
							if i < len(data)-1 {
								embedding[v][data[i+1]]++
							}
						}
						input.Close()
					}
				}
			}
		}
		process("/home/pointlander/projects/testament/golang/go/src/")
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
		output, err := os.Create("embedding.gob")
		if err != nil {
			panic(err)
		}
		defer output.Close()
		encoder := gob.NewEncoder(output)
		err = encoder.Encode(&embedding)
		if err != nil {
			panic(err)
		}
		return
	} else {
		input, err := os.Open("embedding.gob")
		if err != nil {
			panic(err)
		}
		defer input.Close()
		decoder := gob.NewDecoder(input)
		err = decoder.Decode(&embedding)
		if err != nil {
			panic(err)
		}
	}

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
		embedding = [256][256]float32{}
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
		net := NewNet(2, 8, 256, 16)
		in := NewMatrix(0, 256, Batch)
		in.Data = in.Data[:cap(in.Data)]
		position, length := 0, len(data)
		seen := make(map[int]bool, 8)
		for len(seen) != length {
			for i := 0; i < Batch; i++ {
				copy(in.Data[i*256:(i+1)*256], embedding[data[position]][:])
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
			for seen[position] {
				position = (position + 1) % length
			}
			fmt.Println(position, string(data[position]))
		}
		return
	}

	test := func(iterations int) {
		//nets := NewNet(1, 8, 256, 3)
		net := NewNet(2, 8, 256+64, 3)
		in := NewMatrix(0, 256+64, Batch)
		in.Data = in.Data[:cap(in.Data)]
		position := 0
		//rng := rand.New(rand.NewSource(1))
		for position < iterations {
			for i := 0; i < Batch; i++ {
				copy(in.Data[i*256:(i+1)*256], embedding[data[position+i]][:])
			}
			index := uint64(position)
			for i := 0; i < 64; i++ {
				if index&1 == 1 {
					in.Data[256+i] = 1
				} else {
					in.Data[256+i] = 0
				}
				index >>= 1
			}
			//out := nets.Fire(in)
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
