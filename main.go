package main

import (
	"fmt"
	"image/color"
	"math/rand"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

const screenWidth = 1900
const screenHeight = 1000
const numLayers = 5

func drawIntLayer(layer *Layer, x float32, y float32, brushSize float32, brushType string, c color.RGBA) {
	if brushType == "circle" {
		// Draw a circle
		vector.DrawFilledCircle(layer.image, x, y, brushSize, c, true)
	} else {
		// Draw a square
		vector.DrawFilledRect(layer.image, x, y, brushSize, brushSize, c, true)
	}
}

func (g *Game) Update() error {
	// check if w is pressed
	if ebiten.IsKeyPressed(ebiten.KeyW) {
		if !g.keyWPressed {
			if g.currentLayer < numLayers-1 {
				g.currentLayer++
			}
		}
		g.keyWPressed = true
	} else {
		g.keyWPressed = false
	}

	if ebiten.IsKeyPressed(ebiten.KeyS) {
		if !g.keySPressed {
			if g.currentLayer > 0 {
				g.currentLayer--
			}
		}
		g.keySPressed = true
	} else {
		g.keySPressed = false
	}

	// Change the color of the brush
	if ebiten.IsKeyPressed(ebiten.KeyR) {
		g.color = color.RGBA{
			R: uint8(rand.Uint32() & 0xff),
			G: uint8(rand.Uint32() & 0xff),
			B: uint8(rand.Uint32() & 0xff),
			A: uint8(rand.Uint32() & 0xff),
		}
	}



	// increase the brush size by scrolling
	_, scrollY := ebiten.Wheel()
	if scrollY > 0 {
		g.brushSize += 1
	} else if scrollY < 0 {
		// Ensure the brush size does not go below 1
		if g.brushSize > 1 {
			g.brushSize -= 1
		}
	}


	// Draw on the current layer
	if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		x, y := ebiten.CursorPosition()
		drawIntLayer(&g.layers.layers[g.currentLayer], float32(x), float32(y), g.brushSize, "circle", g.color)
	}

	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	// Draw the layers
	for i := 0; i < numLayers; i++ {
		screen.DrawImage(g.layers.layers[i].image, nil)
	}

	// fmt.Printf("FPS: %v", ebiten.ActualFPS())
	ebitenutil.DebugPrint(screen, fmt.Sprintf("FPS: %v", ebiten.ActualFPS()))
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Current Layer: %v", g.currentLayer), 0, 20)
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {
	return 1900, 1000
}

type Layer struct {
	image *ebiten.Image
}

type Layers struct {
	layers [numLayers]Layer
}

type Game struct {
	layers       *Layers
	currentLayer int
	brushSize    float32
	keyWPressed  bool
	keySPressed  bool
	color 	  color.RGBA
}

func main() {
	ebiten.SetVsyncEnabled(false)

	layers := Layers{
		layers: [numLayers]Layer{
			{image: ebiten.NewImage(screenWidth, screenHeight)},
			{image: ebiten.NewImage(screenWidth, screenHeight)},
			{image: ebiten.NewImage(screenWidth, screenHeight)},
			{image: ebiten.NewImage(screenWidth, screenHeight)},
			{image: ebiten.NewImage(screenWidth, screenHeight)},
		},
	}
	// Set the background color of each layer
	for i := 0; i < numLayers; i++ {
		layers.layers[i].image.Fill(color.Transparent)
	}

	game := &Game{
		layers:       &layers,
		currentLayer: 0,
		brushSize:    10,
		color: color.RGBA{255, 0, 0, 255},
	}

	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("Ebiten Game")
	if err := ebiten.RunGame(game); err != nil {
		panic(err)
	}
}
