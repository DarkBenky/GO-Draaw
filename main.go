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
    vector.DrawFilledRect(screen, float32(s.x), float32(s.y), float32(s.w), float32(s.h), color.Gray{0x80} , true)
    vector.DrawFilledRect(screen, float32(s.x)+float32(s.w)*s.value-2, float32(s.y), 4, float32(s.h), s.color , true)
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

	// Draw on the current layer
	if mousePressed {
		drawIntLayer(&g.layers.layers[g.currentLayer], float32(mouseX), float32(mouseY), g.brushSize, g.brushType, g.color)
	}

	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	// Draw the layers
	for i := 0; i < numLayers; i++ {
		screen.DrawImage(g.layers.layers[i].image, nil)
	}

	// Draw button
    vector.DrawFilledRect(screen, float32(g.buttonX), float32(g.buttonY), float32(g.buttonW), float32(g.buttonH), g.color , true)

    // Draw sliders
    for _, slider := range g.sliders {
        slider.Draw(screen)
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
	sliders     [4]*Slider
    buttonX     int
    buttonY     int
    buttonW     int
    buttonH     int
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
		sliders: [4]*Slider{
            {x: 50, y: 350, w: 200, h: 20, color: color.RGBA{255, 0, 0, 255}},
            {x: 50, y: 380, w: 200, h: 20, color: color.RGBA{0, 255, 0, 255}},
            {x: 50, y: 410, w: 200, h: 20, color: color.RGBA{0, 0, 255, 255}},
            {x: 50, y: 440, w: 200, h: 20, color: color.RGBA{0, 0, 0, 255}}, // Alpha slider
        },
        buttonX:     200,
        buttonY:     200,
        buttonW:     100,
        buttonH:     50,
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
