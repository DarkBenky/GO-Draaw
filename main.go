package main

import (
	"fmt"
	"time"
	"image/color"
)

const screenWidth = 600
const screenHeight = 400
const numLayers = 5


type Layer struct {
	mask  [screenWidth][screenHeight]bool 
	color [screenWidth][screenHeight]color.RGBA
}

type Layers struct {
	layers [numLayers]Layer
}

// func addRandomDataToLayer(layer *Layers) {
// 	for row := 0; row < screenWidth; row++ {
// 		for col := 0; col < screenHeight; col++ {
// 			if rand.Intn(2) == 1 {
// 				layer.layers[0].mask[row][col] = true
// 				layer.layers[0].color[row][col] = color.RGBA{0, 0, 0, 255}
// 			}
// 		}
// 	}
// }

func render(layers *Layers) (image [screenWidth][screenHeight]color.RGBA) {
	image = [screenWidth][screenHeight]color.RGBA{}


	for layer := range layers.layers {
		currentLayerColor := &layers.layers[layer].color
		currentLayer := &layers.layers[layer].mask
		for row := range currentLayer{
			currentRow := currentLayer[row]
			for  col := range currentRow{
				if currentRow[col] {
					image[row][col] = currentLayerColor[row][col]
				}
			}
		}
	}
	return image
} 

// func render2(layers *Layers) (image [screenWidth][screenHeight]color.RGBA) {
// 	image = [screenWidth][screenHeight]color.RGBA{}
// 	imageBoll := [screenWidth][screenHeight]bool{}

// 	numLayers := len(layers.layers)
// 	for layer := range layers.layers {
// 		layerReversed := numLayers - layer - 1
// 		currentLayerColor := &layers.layers[layerReversed].color
// 		currentLayer := &layers.layers[layerReversed].mask
// 		for row, currentRow := range currentLayer {
// 			for col, isSet := range currentRow {
// 				if imageBoll[row][col] && isSet {
// 					image[row][col] = currentLayerColor[row][col]
// 					imageBoll[row][col] = true
// 				}
// 			}
// 		}
// 	}
// 	return image
// }


func main() {
	var totalDuration time.Duration
	runs := 5000
	
	// addRandomDataToLayer(&Layers{})

	for i := 0; i < runs; i++ {
		start := time.Now()
		render(&Layers{})
		elapsed := time.Since(start)
		totalDuration += elapsed
	}

	averageTime := totalDuration / time.Duration(runs)
	fmt.Printf("Average time: %v\n", averageTime)
	fmt.Printf("Average FPS: %v\n", 1e9/averageTime.Nanoseconds())

	// for i := 0; i < runs; i++ {
	// 	start := time.Now()
	// 	render2(&Layers{})
	// 	elapsed := time.Since(start)
	// 	totalDuration += elapsed
	// }

	// averageTime = totalDuration / time.Duration(runs)
	// fmt.Printf("Average time V2: %v\n", averageTime)
	// fmt.Printf("Average FPS V2: %v\n", 1e9/averageTime.Nanoseconds())
}
