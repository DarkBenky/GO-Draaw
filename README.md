# GO-Draaw
GO-Draaw is a project written in Go that allows users to draw on multiple layers using the Ebiten library. The program provides a canvas with adjustable brush size, brush type (circle or square), and color. Users can switch between layers, change brush properties, and draw on the canvas using the mouse. The program also includes sliders to adjust the RGB values of the brush color. Additionally, there is a button on the canvas. The program uses the Ebiten library for graphics and input handling.

## Testing 
go tool pprof http://localhost:6060/debug/pprof/profile\?seconds\=30
## Show 
go tool pprof -http=:8080 pprof.main.samples.cpu.007.pb