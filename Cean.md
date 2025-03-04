## 2.1 Architektúra Projektu GO-Draaw
Projekt je rozdelený na dve hlavné časti:
* Frontend: Vue.js
* Backend: Golang with Echo Framework a RayTracer

## 2.2 Backend Implementácia Web Servera
### 2.2.1 Koncové body
* `POST /submitColor`: Odoslať farebné údaje
```go
Color struct {
    R               float64 `json:"r"`
    G               float64 `json:"g"`
    B               float64 `json:"b"`
    A               float64 `json:"a"`
    Reflection      float64 `json:"reflection"`
    Roughness       float64 `json:"roughness"`
    DirectToScatter float64 `json:"directToScatter"`
    Metallic        float64 `json:"metalic"`
    RenderVolume    bool    `json:"renderVolume"`
    RenderVoxels    bool    `json:"renderVoxels"`
}
```

* `POST /submitVoxel`: Odoslať voxel údaje
```go
type Volume struct {
    Density               float64 `json:"density"`
    Transmittance         float64 `json:"transmittance"`
    Randomnes             float64 `json:"randomness"`
    SmokeColorR           float64 `json:"smokeColorR"`
    SmokeColorG           float64 `json:"smokeColorG"`
    SmokeColorB           float64 `json:"smokeColorB"`
    SmokeColorA           float64 `json:"smokeColorA"`
    VoxelColorR           float64 `json:"voxelColorR"`
    VoxelColorG           float64 `json:"voxelColorG"`
    VoxelColorB           float64 `json:"voxelColorB"`
    VoxelColorA           float64 `json:"voxelColorA"`
    RandomnessVoxel       float64 `json:"randomnessVoxel"`
    RenderVolume          bool    `json:"renderVolume"`
    RenderVoxel           bool    `json:"renderVoxel"`
    OverWriteVoxel        bool    `json:"overWriteVoxel"`
    VoxelModification     string  `json:"voxelModification"`
    UseRandomnessForPaint bool    `json:"useRandomnessForPaint"`
    ConvertVoxelsToSmoke  bool    `json:"convertVoxelsToSmoke"`
}
```

* `POST /submitTextures`: Odoslať textúrové údaje
```go
type TextureRequest struct {
    Textures          map[string]interface{} `json:"textures"`
    Normals           map[string]interface{} `json:"normals"`
    DirectToScatter   float64                `json:"directToScatter"`
    Reflection        float64                `json:"reflection"`
    Roughness         float64                `json:"roughness"`
    Metallic          float64                `json:"metallic"`
    Index             int                    `json:"index"`
    Specular          float64                `json:"specular"`
    ColorR            float64                `json:"colorR"`
    ColorG            float64                `json:"colorG"`
    ColorB            float64                `json:"colorB"`
    ColorA            float64                `json:"colorA"`
}
```

* `POST /submitRenderOptions`: Odoslať konfiguráciu renderingu
```go
type RenderOptions struct {
    Depth            int     `json:"depth"`
    Scatter          int     `json:"scatter"`
    Gamma            float64 `json:"gamma"`
    SnapLight        string  `json:"snapLight"`
    RayMarching      string  `json:"rayMarching"`
    Performance      string  `json:"performance"`
    Mode             string  `json:"mode"`
    Resolution       string  `json:"resolution"`
    Version          string  `json:"version"`
    FOV              float64 `json:"fov"`
    LightIntensity   float64 `json:"lightIntensity"`
    R                float64 `json:"r"`
    G                float64 `json:"g"`
    B                float64 `json:"b"`
}
```

* `POST /submitShader`: Odoslať konfiguráciu shadera
```go
type ShaderParam struct {
    Type       string                 `json:"type"`
    Parameters map[string]interface{} `json:"params"`
}
```

* `GET /getCameraPosition`: Získať aktuálnu pozíciu kamery
```go
type Position struct {
    X       float64 `json:"x"`
    Y       float64 `json:"y"`
    Z       float64 `json:"z"`
    CameraX float64 `json:"cameraX"`
    CameraY float64 `json:"cameraY"`
}
```

* `POST /moveToPosition`: Presunúť kameru na určenú pozíciu
```go
type Position struct {
    X       float64 `json:"x"`
    Y       float64 `json:"y"`
    Z       float64 `json:"z"`
    CameraX float64 `json:"cameraX"`
    CameraY float64 `json:"cameraY"`
}
```

### 2.2.2 Backend Architektúra
Backend je asynchrónny a beží v samostatnom Go routine. Využíva unsafe funkcionality Go na vyhnutie sa komplexnostiam spravovania stavu a réžii mutexov, čo vedie k zlepšenému výkonu.