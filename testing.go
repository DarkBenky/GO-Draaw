package main

import (
	"fmt"
	"log"
	"strings"

	"github.com/go-gl/gl/v4.1-core/gl"
	"github.com/go-gl/glfw/v3.3/glfw"
)

var vertexShader = `
#version 410 core
layout (location = 0) in vec2 aPos;
out vec2 TexCoords;

void main()
{
    TexCoords = aPos * 0.5 + 0.5;
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0); 
}
` + "\x00"

var fragmentShader = `
#version 410 core
out vec4 FragColor;
in vec2 TexCoords;

uniform vec3 rayOrigin;
uniform vec3 rayDir;
uniform vec3 boxMin;
uniform vec3 boxMax;
uniform float aspectRatio;

bool rayBoxIntersect(vec3 origin, vec3 dir, vec3 boxMin, vec3 boxMax, out float t) {
    vec3 invDir = 1.0 / dir;
    vec3 tMin = (boxMin - origin) * invDir;
    vec3 tMax = (boxMax - origin) * invDir;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);

    if (tNear > tFar || tFar < 0.0) {
        return false;
    }

    t = tNear;
    return true;
}

void main() {
    // Calculate ray direction based on pixel position
    vec2 uv = TexCoords * 2.0 - 1.0;
    uv.x *= aspectRatio;
    vec3 rayDir = normalize(vec3(uv, -1.0));

    float t;
    bool hit = rayBoxIntersect(rayOrigin, rayDir, boxMin, boxMax, t);

    if (hit) {
        vec3 hitPoint = rayOrigin + t * rayDir;
        vec3 normal = sign(hitPoint - (boxMin + boxMax) * 0.5);
        vec3 color = abs(normal);
        FragColor = vec4(color, 1.0);
    } else {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}
` + "\x00"

func compileShader(source string, shaderType uint32) (uint32, error) {
	shader := gl.CreateShader(shaderType)
	csources, free := gl.Strs(source)
	gl.ShaderSource(shader, 1, csources, nil)
	free()
	gl.CompileShader(shader)

	var status int32
	gl.GetShaderiv(shader, gl.COMPILE_STATUS, &status)
	if status == gl.FALSE {
		var logLength int32
		gl.GetShaderiv(shader, gl.INFO_LOG_LENGTH, &logLength)

		log := strings.Repeat("\x00", int(logLength+1))
		gl.GetShaderInfoLog(shader, logLength, nil, gl.Str(log))

		return 0, fmt.Errorf("failed to compile %v: %v", source, log)
	}

	return shader, nil
}

func initOpenGL() uint32 {
	if err := gl.Init(); err != nil {
		panic(err)
	}
	version := gl.GoStr(gl.GetString(gl.VERSION))
	fmt.Println("OpenGL version", version)

	vertexShader, err := compileShader(vertexShader, gl.VERTEX_SHADER)
	if err != nil {
		log.Fatalln(err)
	}

	fragmentShader, err := compileShader(fragmentShader, gl.FRAGMENT_SHADER)
	if err != nil {
		log.Fatalln(err)
	}

	prog := gl.CreateProgram()
	gl.AttachShader(prog, vertexShader)
	gl.AttachShader(prog, fragmentShader)
	gl.LinkProgram(prog)

	var status int32
	gl.GetProgramiv(prog, gl.LINK_STATUS, &status)
	if status == gl.FALSE {
		var logLength int32
		gl.GetProgramiv(prog, gl.INFO_LOG_LENGTH, &logLength)

		log := strings.Repeat("\x00", int(logLength+1))
		gl.GetProgramInfoLog(prog, logLength, nil, gl.Str(log))

		fmt.Println("failed to link program:", log)
	}

	gl.DeleteShader(vertexShader)
	gl.DeleteShader(fragmentShader)

	gl.UseProgram(prog)

	return prog
}

func main() {
    // Initialize GLFW
    if err := glfw.Init(); err != nil {
        panic(err)
    }
    defer glfw.Terminate()

    glfw.WindowHint(glfw.ContextVersionMajor, 4)
    glfw.WindowHint(glfw.ContextVersionMinor, 1)
    glfw.WindowHint(glfw.OpenGLProfile, glfw.OpenGLCoreProfile)
    glfw.WindowHint(glfw.OpenGLForwardCompatible, glfw.True)

    window, err := glfw.CreateWindow(800, 600, "Ray-Box Collision on GPU", nil, nil)
    if err != nil {
        panic(err)
    }
    window.MakeContextCurrent()

    program := initOpenGL()

    var vao uint32
    gl.GenVertexArrays(1, &vao)
    gl.BindVertexArray(vao)

    var vbo uint32
    gl.GenBuffers(1, &vbo)
    gl.BindBuffer(gl.ARRAY_BUFFER, vbo)

    quadVertices := []float32{
        -1.0, 1.0,
        -1.0, -1.0,
        1.0, -1.0,
        -1.0, 1.0,
        1.0, -1.0,
        1.0, 1.0,
    }
    gl.BufferData(gl.ARRAY_BUFFER, len(quadVertices)*4, gl.Ptr(quadVertices), gl.STATIC_DRAW)

    gl.EnableVertexAttribArray(0)
    gl.VertexAttribPointer(0, 2, gl.FLOAT, false, 2*4, gl.PtrOffset(0))

    // Ray data
    rayOrigin := [3]float32{0, 0, -5}

    // Box data
    boxMin := [3]float32{-1, -1, -1}
    boxMax := [3]float32{1, 1, 1}

    width, height := window.GetSize()
    aspectRatio := float32(width) / float32(height)

    // Get uniform locations
    rayOriginUniform := gl.GetUniformLocation(program, gl.Str("rayOrigin\x00"))
    if rayOriginUniform == -1 {
        log.Println("Warning: rayOrigin uniform not found")
    }

    boxMinUniform := gl.GetUniformLocation(program, gl.Str("boxMin\x00"))
    if boxMinUniform == -1 {
        log.Println("Warning: boxMin uniform not found")
    }

    boxMaxUniform := gl.GetUniformLocation(program, gl.Str("boxMax\x00"))
    if boxMaxUniform == -1 {
        log.Println("Warning: boxMax uniform not found")
    }

    aspectRatioUniform := gl.GetUniformLocation(program, gl.Str("aspectRatio\x00"))
    if aspectRatioUniform == -1 {
        log.Println("Warning: aspectRatio uniform not found")
    }

    for !window.ShouldClose() {
        gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

        // Pass data to shader
        if rayOriginUniform != -1 {
            gl.Uniform3f(rayOriginUniform, rayOrigin[0], rayOrigin[1], rayOrigin[2])
        }
        if boxMinUniform != -1 {
            gl.Uniform3f(boxMinUniform, boxMin[0], boxMin[1], boxMin[2])
        }
        if boxMaxUniform != -1 {
            gl.Uniform3f(boxMaxUniform, boxMax[0], boxMax[1], boxMax[2])
        }
        if aspectRatioUniform != -1 {
            gl.Uniform1f(aspectRatioUniform, aspectRatio)
        }

        gl.DrawArrays(gl.TRIANGLES, 0, 6)

        window.SwapBuffers()
        glfw.PollEvents()
    }
}
