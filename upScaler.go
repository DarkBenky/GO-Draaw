package main

import (
	"fmt"
	"image"
	"os"
	"path/filepath"

	"github.com/aunum/goro/pkg/v1/layer"
	"github.com/aunum/goro/pkg/v1/model"
	"gorgonia.org/tensor"
	"github.com/nfnt/resize"
	"gorgonia.org/gorgonia" // Added for g.NewAdamSolver()
)

func loadImage(filePath string, size image.Point) (*tensor.Dense, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}

	img = resizeImage(img, size)

	imgTensor := imageToTensor(img)
	return imgTensor, nil
}

func resizeImage(img image.Image, size image.Point) image.Image {
	return resize.Resize(uint(size.X), uint(size.Y), img, resize.Lanczos3)
}

func imageToTensor(img image.Image) *tensor.Dense {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	data := make([]float32, width*height)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			// Convert to grayscale and normalize to [0, 1]
			data[y*width+x] = float32(0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)) / 65535
		}
	}

	return tensor.New(tensor.WithShape(1, height, width), tensor.WithBacking(data))
}

func loadDataset(dir string, size image.Point) ([]*tensor.Dense, []*tensor.Dense, error) {
	var inputs []*tensor.Dense
	var labels []*tensor.Dense

	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if !info.IsDir() {
			imgTensor, err := loadImage(path, size)
			if err != nil {
				return err
			}

			inputs = append(inputs, imgTensor)
			// For simplicity, we assume labels are the same as inputs
			labels = append(labels, imgTensor)
		}
		return nil
	})

	if err != nil {
		return nil, nil, err
	}

	return inputs, labels, nil
}

func main() {
	// Define image sizes
	inputSize := image.Point{X: 64, Y: 64}
	outputSize := image.Point{X: 128, Y: 128}

	// Load dataset
	inputs, labels, err := loadDataset("/home/user/Downloads/Linnaeus 5 128X128/train/other", inputSize)
	if err != nil {
		fmt.Println("Error loading dataset:", err)
		return
	}

	// Create the 'x' input e.g. low-resolution image
	x := model.NewInput("x", []int{1, inputSize.Y, inputSize.X})

	// Create the 'y' or expected output e.g. high-resolution image
	y := model.NewInput("y", []int{1, outputSize.Y, outputSize.X})

	// Create a new sequential model with the name 'upscale'
	model, err := model.NewSequential("upscale")
	if err != nil {
		fmt.Println("Error creating model:", err)
		return
	}

	// Add layers to the model
	err = model.AddLayers(
		layer.Conv2D{Input: 1, Output: 64, Width: 5, Height: 5, Padding: "same"},
		layer.ReLU{},
		layer.Conv2D{Input: 64, Output: 64, Width: 5, Height: 5, Padding: "same"},
		layer.ReLU{},
		layer.UpSampling2D{Scale: 2},
		layer.Conv2D{Input: 64, Output: 32, Width: 5, Height: 5, Padding: "same"},
		layer.ReLU{},
		layer.Conv2D{Input: 32, Output: 1, Width: 5, Height: 5, Padding: "same"},
	)
	if err != nil {
		fmt.Println("Error adding layers:", err)
		return
	}

	// Pick an optimizer
	optimizer := gorgonia.NewAdamSolver()

	// Compile the model with options
	err = model.Compile(x, y,
		model.WithOptimizer(optimizer),
		model.WithLoss(model.MeanSquaredError),
		model.WithBatchSize(16),
	)
	if err != nil {
		fmt.Println("Error compiling model:", err)
		return
	}

	// Fit the model
	err = model.Fit(inputs, labels)
	if err != nil {
		fmt.Println("Error fitting model:", err)
		return
	}
}