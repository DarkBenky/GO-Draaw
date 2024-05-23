// FIXME[High]: UI elements merge into one layer and then draw on the screen
// [DONE]: Implement the blur function 
// TODO [High]: Implement the increase contrast function
// TODO [High]: Implement the increase brightness function
// TODO [High]: Implement the decrease brightness function
// TODO [High]: Implement the edge detection function
// TODO [High]: Implement the decrease contrast function
// TODO [High]: Implement the sharpen function
// TODO [High]: Implement the eraser tool
// TODO [High]: Implement the fill tool
// TODO [High]: Implement the line tool

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

const maxMaskSize = 100

type Mask struct {
	mask            [maxMaskSize][maxMaskSize]color.RGBA
	currentMaskSize int
}

func (mask *Mask) createMask(layer *Layer, x int, y int, g *Game) {
	brushSize := g.brushSize
	mask.currentMaskSize = int(brushSize)
	if brushSize > maxMaskSize {
		brushSize = maxMaskSize
	}
	for i := 0; i < int(brushSize); i++ {
		for j := 0; j < int(brushSize); j++ {
			newX := x - int(brushSize)/2 + i
			newY := y - int(brushSize)/2 + j
			if newX >= 0 && newX < screenWidth && newY >= 0 && newY < screenHeight {
				mask.mask[i][j] = layer.image.At(newX, newY).(color.RGBA)
			}
		}
	}
}

type ColorInt16 struct {
	R uint16
	G uint16
	B uint16
	A uint16
}

// blurLayer applies a blur effect to a specific region of the layer's image.
// It takes the x and y coordinates of the region, a pointer to the Game struct,
// and a pointer to the Mask struct as parameters.
// The function creates a mask using the current layer, x, y, and game parameters.
// It then applies the blur effect to the region defined by the mask.
// The blur effect is calculated by averaging the color values of the pixels
// within a kernel size determined by the brush size of the game.
// The resulting blurred pixels are stored in a temporary matrix.
// Finally, the blurred pixels are applied to the layer's image at the
// corresponding positions defined by the mask.
func (layer *Layer) blurLayer(x int, y int, game *Game, mask *Mask) {
	mask.createMask(layer, x, y, game)
	temp := make([][]color.RGBA, mask.currentMaskSize)
	for i := range temp {
		temp[i] = make([]color.RGBA, mask.currentMaskSize)
	}

	kernelSize :=  int(game.brushSize / 25) + 1

	for h := 0; h < mask.currentMaskSize; h++ {
		for w := 0; w < mask.currentMaskSize; w++ {
			average := ColorInt16{}
			count := uint16(0)
			for i := -kernelSize; i <= kernelSize; i++ {
				for j := -kernelSize; j <= kernelSize; j++ {
					nh, nw := h+i, w+j
					if nh >= 0 && nh < mask.currentMaskSize && nw >= 0 && nw < mask.currentMaskSize {
						average.R += uint16(mask.mask[nh][nw].R)
						average.G += uint16(mask.mask[nh][nw].G)
						average.B += uint16(mask.mask[nh][nw].B)
						average.A += uint16(mask.mask[nh][nw].A)
						count++
					}
				}
			}
			average.R /= count
			average.G /= count
			average.B /= count
			average.A /= count
			temp[h][w] = color.RGBA{uint8(average.R), uint8(average.G), uint8(average.B), uint8(average.A)}
		}
	}

	for h := 0; h < mask.currentMaskSize; h++ {
		for w := 0; w < mask.currentMaskSize; w++ {
			newX := x - mask.currentMaskSize/2 + h
			newY := y - mask.currentMaskSize/2 + w
			if newX >= 0 && newX < screenWidth && newY >= 0 && newY < screenHeight {
				layer.image.Set(newX, newY, temp[h][w])
			}
		}
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
		g.layers.layers[g.currentLayer].blurLayer(mouseX, mouseY, g, &g.mask)
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
	mask 		Mask
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
		mask : Mask{},
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
