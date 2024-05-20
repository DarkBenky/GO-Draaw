package main

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

const screenWidth = 1900
const screenHeight = 1000
const numLayers = 5

type Button struct {
	x, y, w, h  int
	text        string
	background  color.RGBA
	valueReturn int
}

func (b *Button) Draw(screen *ebiten.Image) {
	vector.DrawFilledRect(screen, float32(b.x), float32(b.y), float32(b.w), float32(b.h), b.background, true)
	ebitenutil.DebugPrintAt(screen, b.text, b.x, b.y)
}

type Slider struct {
	x, y, w, h int
	value      float32
	color      color.RGBA
}

func (s *Slider) Update(mouseX, mouseY int, mousePressed bool) {
	if mousePressed && mouseX >= s.x && mouseX <= s.x+s.w && mouseY >= s.y && mouseY <= s.y+s.h {
		s.value = float32(mouseX-s.x) / float32(s.w)
	}
}

func (s *Slider) Draw(screen *ebiten.Image) {
	vector.DrawFilledRect(screen, float32(s.x), float32(s.y), float32(s.w), float32(s.h), color.Gray{0x80}, true)
	vector.DrawFilledRect(screen, float32(s.x)+float32(s.w)*s.value-2, float32(s.y), 4, float32(s.h), s.color, true)
}

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

func drawIntLayer(layer *Layer, x float32, y float32, g *Game) {
	if g.brushType == 0 {
		// Draw a circle
		vector.DrawFilledCircle(layer.image, x, y, g.brushSize, g.color, true)
	} else {
		// Draw a square
		vector.DrawFilledRect(layer.image, x, y, g.brushSize, g.brushSize, g.color, true)
	}
}

// TODO: optimize this function
func blurIntLayer(layer *Layer, x, y int, game *Game) {
	
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

	mouseX, mouseY := ebiten.CursorPosition()
	mousePressed := ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft)

	for _, slider := range g.sliders {
		slider.Update(mouseX, mouseY, mousePressed)
	}

	g.color = color.RGBA{
		uint8(g.sliders[0].value * 255),
		uint8(g.sliders[1].value * 255),
		uint8(g.sliders[2].value * 255),
		uint8(g.sliders[3].value * 255),
	}

	// Get Button Clicks
	if mousePressed {
		for _, button := range g.Buttons {
			if mouseX >= button.x && mouseX <= button.x+button.w && mouseY >= button.y && mouseY <= button.y+button.h {
				g.currentTool = button.valueReturn
			}
		}
	}

	// Draw on the current layer
	if mousePressed && g.currentTool == 0 {
		drawIntLayer(&g.layers.layers[g.currentLayer], float32(mouseX), float32(mouseY), g)
	} else if mousePressed && g.currentTool == 1 {
		blurIntLayer(&g.layers.layers[g.currentLayer], mouseX, mouseY, g)
	}


	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	// Draw the layers
	for i := 0; i < numLayers; i++ {
		screen.DrawImage(g.layers.layers[i].image, nil)
	}

	// Draw sliders
	for _, slider := range g.sliders {
		slider.Draw(screen)
	}
	// Draw current color preview
	vector.DrawFilledRect(screen, 50, 470, 200, 200, g.color, true)

	// Draw buttons
	for _, button := range g.Buttons {
		button.Draw(screen)
	}

	// fmt.Printf("FPS: %v", ebiten.ActualFPS())
	ebitenutil.DebugPrint(screen, fmt.Sprintf("FPS: %v", ebiten.ActualFPS()))
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Current Layer: %v", g.currentLayer), 0, 20)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Brush Size: %v", g.brushSize), 0, 40)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Brush Type: %v", g.brushType), 0, 60)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Current Tool: %v", g.currentTool), 0, 80)
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
	layers        *Layers
	currentLayer  int
	brushSize     float32
	brushType     int
	prevKeyStates map[ebiten.Key]bool
	currKeyStates map[ebiten.Key]bool
	color         color.RGBA
	sliders       [4]*Slider
	Buttons       []*Button
	currentTool   int
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
	buttons := []*Button{
		{x: 200, y: 200, w: 100, h: 50, text: "Button 1", background: color.RGBA{255, 0, 0, 0}, valueReturn: 0},
		{x: 200, y: 300, w: 100, h: 50, text: "Button 2", background: color.RGBA{0, 255, 0, 255}, valueReturn: 1},
		{x: 200, y: 400, w: 100, h: 50, text: "Button 3", background: color.RGBA{0, 0, 255, 255}, valueReturn: 2},
	}

	// Set the background color of each layer
	for i := 0; i < numLayers; i++ {
		layers.layers[i].image.Fill(color.Transparent)
	}

	game := &Game{
		layers:        &layers,
		currentLayer:  0,
		brushSize:     10,
		color:         color.RGBA{255, 0, 0, 255},
		brushType:     0,
		prevKeyStates: make(map[ebiten.Key]bool),
		currKeyStates: make(map[ebiten.Key]bool),
		sliders: [4]*Slider{
			{x: 50, y: 350, w: 200, h: 20, color: color.RGBA{255, 0, 0, 255}},
			{x: 50, y: 380, w: 200, h: 20, color: color.RGBA{0, 255, 0, 255}},
			{x: 50, y: 410, w: 200, h: 20, color: color.RGBA{0, 0, 255, 255}},
			{x: 50, y: 440, w: 200, h: 20, color: color.RGBA{0, 0, 0, 255}}, // Alpha slider
		},
		currentTool: 0,
		Buttons:     buttons,
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
