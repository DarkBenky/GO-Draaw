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

func (g *Game) isKeyReleased(key ebiten.Key) bool {
	// Check if the key was previously pressed and is now released
	return g.prevKeyStates[key] && !g.currKeyStates[key]
}

func (g *Game) updateKeyStates() {
	// Update the current key states
	for k := range g.currKeyStates {
		g.currKeyStates[k] = ebiten.IsKeyPressed(k)
	}
}

func (g *Game) storePrevKeyStates() {
	// Store the current key states as the previous key states
	for k, v := range g.currKeyStates {
		g.prevKeyStates[k] = v
	}
}

func drawIntLayer(layer *Layer, x float32, y float32, brushSize float32, brushType int, c color.RGBA) {
	if brushType == 0 {
		// Draw a circle
		vector.DrawFilledCircle(layer.image, x, y, brushSize, c, true)
	} else {
		// Draw a square
		vector.DrawFilledRect(layer.image, x, y, brushSize, brushSize, c, true)
	}
}

func (g *Game) Update() error {
	// Update the current key states
	g.updateKeyStates()

	if g.isKeyReleased(ebiten.KeyW) {
		if g.currentLayer < numLayers-1 {
			g.currentLayer++
		}
	}

	if g.isKeyReleased(ebiten.KeyS) {
		if g.currentLayer > 0 {
			g.currentLayer--
		}
	}

	if g.isKeyReleased(ebiten.KeyQ) {
		g.brushType = (g.brushType + 1) % 2 // Toggle between 0 and 1
	}

	if g.isKeyReleased(ebiten.KeyR) {
		g.color = color.RGBA{uint8(rand.Intn(255)), uint8(rand.Intn(255)), uint8(rand.Intn(255)), 255}
	}

	// Store the current key states as the previous states for the next update
	g.storePrevKeyStates()

	// increase the brush size by scrolling
	_, scrollY := ebiten.Wheel()
	if scrollY > 0 {
		g.brushSize += 1
	} else if scrollY < 0 {
		g.brushSize -= 1
	}

	


	// Draw on the current layer
	if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		x, y := ebiten.CursorPosition()
		drawIntLayer(&g.layers.layers[g.currentLayer], float32(x), float32(y), g.brushSize, g.brushType, g.color)
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
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Brush Size: %v", g.brushSize), 0, 40)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Brush Type: %v", g.brushType), 0, 60)
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
	brushType    int
	prevKeyStates map[ebiten.Key]bool
	currKeyStates map[ebiten.Key]bool
	color 	  color.RGBA
}

func main() {
	ebiten.SetVsyncEnabled(false)

	ebiten.SetTPS(60)

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
		brushType: 0,
		prevKeyStates: make(map[ebiten.Key]bool),
		currKeyStates: make(map[ebiten.Key]bool),
	}

	keys := []ebiten.Key{ebiten.KeyW, ebiten.KeyS, ebiten.KeyQ, ebiten.KeyR}
	for _, key := range keys {
		game.prevKeyStates[key] = false
		game.currKeyStates[key] = false
	}

	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("Ebiten Game")
	if err := ebiten.RunGame(game); err != nil {
		panic(err)
	}
}
