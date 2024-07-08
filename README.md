# GO-Draaw
GO-Draaw is a project written in Go that allows users to draw on multiple layers using the Ebiten library. The program provides a canvas with adjustable brush size, brush type (circle or square), and color. Users can switch between layers, change brush properties, and draw on the canvas using the mouse. The program also includes sliders to adjust the RGB values of the brush color. Additionally, there is a button on the canvas. The program uses the Ebiten library for graphics and input handling.

## run GO => LD_LIBRARY_PATH=${PWD}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} go run main.go

## compile CUDA => nvcc --ptxas-options=-v --compiler-options '-fPIC' -o libmaxmul.so --shared maxmul.cu
