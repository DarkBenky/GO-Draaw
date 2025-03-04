Obsah

1.0 Úvod
2.1 Architektúra Aplikácie
    - RayTracingEngine
    - UI 
2.1.1 Princíp fungovania BVH
    - Rozdiel Medzi BVH Node a BVHLean
2.1.2 Surface Area Heuristic (SAH)
2.1.3 Implementačné prístupy
2.2 Reprezentácia Trojuholníkov a Materiálové Vlastnosti
    - Rozdiel Medzi Lean verziu a Standar verziou
2.2.1 Geometrická Reprezentácia
2.2.2 Materiálové Vlastnosti
2.3 Podpora Načítavania 3D Geometrie
2.3.1 Načítavanie .OBJ Súborov
2.3.2 Ukladanie a Načítavanie BVH Štruktúr
2.4 Podpora Post-Processing Shaderov
    - Rozpis shadrov
2.4.1 Charakteristika Kage Shader Language
2.4.2 Implementované Post-Processing Efekty
2.4.3 Architektúra Podpory Shaderov 
2.5 Systém Benchmarkovania a Výkonnostnej Analýzy
2.6 Volume Rendering
2.7 Voxel Redering
    - Methody pre Upravu Voxelov 
    - Converzia Voxelov Na Volume
Záver
Zdroje

UI
    Technicke Parametre 
        Vue js a Golang backend v Echo frameworku
    Backend Implementacia
        endpointy 
        e.POST("/submitColor", game.submitColor)
	    e.POST("/submitVoxel", game.submitVoxelData)
	    e.POST("/submitRenderOptions", game.submitRenderOptions)
	    e.POST("/submitTextures", game.submitTextures)
	    e.POST("/submitShader", game.SubmitShader)
	    e.GET("/getCameraPosition", game.GetPositions)
	    e.POST("/moveToPosition", game.MoveToCameraPosition)
        
        backend je asynchroni a bezi v rozdielnej go rutine a vyziva usefe funkcionalitu v golagu aby sa vyhlo state mangmentu a mutexom taktiez to ma za dosledok lepsi vykon
    Casti vo Frontende 
        Color Picker
            R 0-256 * multiplayer
            G 0-256 * multiplayer
            B 0-256 * multiplayer
            A 0-256 * multiplayer
            
            Preview 
            Ukazka akuratnej farby ktora je selctnuta

            v tejto casti je mozne zadat farbu ktora bude uplatnena pre trojuholnik na ktory je kliknuty 

        Texture Color Picker
            Texture Selector
                obsahuje selector medzi 1 128 v ktorej je mozne vybrat jednu z textur
            Texture Preview 
                Textura je v rozliseni 128 * 128 * 4 float

                Submit Textures Button pre Nahratie textury

                v danej casti je mozne zobrazit texturu a nasledne ju editovan na zaklade vybranj farby z Color Pickeru 
            Normal Preview 
                Normal map je v rozliseni v 128 *128 * 3 
                normal map je noramlizovan v rozasuj -1 - 1 
                noraml map je nasledne na backend konvetovany na vector

                Upload Normal Button sluzi na nahranie normal map

                No noraml Button otvori stranku https://cpetry.github.io/NormalMap-Online/ kde je mozne vytvorit normal map z obrazku

                V pravej casti su vypisane aktualne hodnoty pre pre material texturu Reflection Direct to Scatter Roughness Metallic Specular

            Submit Textures
                nahraje oznacenu texturu na backend

            Addition Setting
                slider pre Direct to Scatter Roughness Metallic Specular v rozsahu 0-1

                Red/ Green / Blue / Alpa chanel multiplayer upravy texturu a vynasoby ju danou hondnotou

        Shader Menu 
            sluzi na vytvoreniooe postprocesing shadrow ktore mozu byt postupne za sebou ako napr original image => contrast => tint => final image

            selector 
                v danej casty je mozne vybrat jeden z post procesing shadrow
            
            add shader Button
                prida novy shader na zaklade vybraneho selectora

            submit shder menu button 
                sluzi pre nahranie shadrow na backend

            shadre a parametre

                ammout hovor aky pomer upraveneho ubrazku sa ma pridat do rendra

                multipass hovor kolko razy za sebou sa ma uplatnit dany post procesing shader za sebou napr multipass 2 
                    image => shader modiefied image => shader modiefied image => new image


                Contrast

                    definia ...

                    ammount
                    multipass
                    Contrast - sila filtra
                Tint

                    definia ...

                    ammount
                    multipass
                    tint color
                    tint strengt hovori ako silno ma byt uplatneni tint shader
                Bloom

                    definia ...

                    ammount
                    multipass
                    treshold hovori hranicu po ktorej je uplatneny bloom shader
                    Intensity sila bloom shadra
                BloomV2

                    definia ...

                    ammount
                    multipass
                    treshold hovori hranicu po ktorej je uplatneny bloom shader
                    Intensity sila bloom shadra
                sharpnes

                    definia ...

                    ammount
                    multipass
                    sharpnes - sila filtra
                ColorMapping

                    definicia  Definuje kolko farieb je pre dany kanal ak je napr 2 tak but je to 0% alebo 100%

                    ammount
                    multipass
                    ColorR
                    ColorG
                    ColorB
                Chromatic Aboration

                    definicia uplatnuje posunutie farebneho kanalu red (lavo) blue (pravo)

                    ammount
                    multipass
                    strenght - sila filtra

                Edge Detection 

                    definicia poziva sobel filter na zvyracnenie egov a akej farby amju byt tieto edge zviraznene

                    ammount
                    multipass
                    strenght - sila filtra

                    Color R
                    Color G
                    Color B
                
                Lighten 

                    definicia upratnuje multiplayer ktory zosvetli dany obrazok

                     ammount
                    multipass
                    strenght - sila filtra

        Render Otions 
            Submit Render Options
                sluzi na nahratie nastaveny
            
            Get Camera position Button
                 dostane poziciu kamery a jej rotaciu

            Hide / Show Camera Position 
                dane menu umoznuje zobrazenie pozicii kamery a umoznuje presunut kameru na danu poziciu

            
            Main Parameters
                Depht - kolko odrazov ma renderer zobrazit
                Scatter - kolko lucov ma byt odrazenych roznimiu smermi z povrchu pre lepsi detail je vyhodne zvisit hodnotu
                
                Lighting Paramters
                    definicia sila a farba svetla

                    Light Intensity sila svetla
                    R farba svetla
                    G farba svetla 
                    B farba svetla
                
                Field of view rozsah viditeľnej scény\
                Gama kontrast a jas medzi tmavými a svetlými tónmi.

            
            Render Setting
                Snap Light To Camera da svetlo na pozicu cameri
                Raymarching Aktualne neimplementovany 
                Preformance Mode zrusi wg.Wait v dosledku je obrazok trhanejsi ale vyuziva sa cely vykon hardveru
                Mode 
                    Clasic - noramlne rendrovanie
                    Normal - rendrovanie normalu pre obrazok len pre V2Log V2Lin V2LogTexture V2LinTexture V4Log V4Lin
                    Distance - aktualne nieje implementovane korektne

        Volume Picker 
            Submit Volume colors nahratie na backend
            Color randomnes uplatnuje nahodnu zlozku na vybratu farbu
            Render Voxels Definuje ci sa maju rendrovat voxeli
            OverWrite Voxels ak je vybrata tatoi moznost vsetky voxeli zmenia svoju farbu
            Add Randomnes to painting ak je tato moznots vybrata je pridana nahodna zlozka do farby ktora je pozivana pri uprave voxelov
            Convert Voxels to smoke prepise hodnoti z voxelov vo voxelov pre Volume (smoke glass etc)

            Voxel color 
                Slector na vybratie farby voxelov s Preview
            Color randomnes Pre Volume
            Render Volume definuje ci ma byt dany element rendrovany
            Smoke Color 
                sluzy na vybratie farby pre volume a obasahuje preview danej farby
            Volume Properties
                Density hustota
                Transmitance definuje priehladnot volumu
                    





## Technical Parameters
- Frontend: Vue.js
- Backend: Golang with Echo Framework

## Backend Implementation
### Endpoints
- `POST /submitColor`: Submit color data
type Color struct {
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
- `POST /submitVoxel`: Submit voxel data
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

- `POST /submitRenderOptions`: Submit render configuration
    type RenderOptions struct {
		Depth          int     `json:"depth"`
		Scatter        int     `json:"scatter"`
		Gamma          float64 `json:"gamma"`
		SnapLight      string  `json:"snapLight"`
		RayMarching    string  `json:"rayMarching"`
		Performance    string  `json:"performance"`
		Mode           string  `json:"mode"`
		Resolution     string  `json:"resolution"`
		Version        string  `json:"version"`
		FOV            float64 `json:"fov"`
		LightIntensity float64 `json:"lightIntensity"`
		R              float64 `json:"r"`
		G              float64 `json:"g"`
		B              float64 `json:"b"`
	}
- `POST /submitTextures`: Submit texture data
    type TextureRequest struct {
		Textures map[string]interface{} `json:"textures"`
		Normals  map[string]interface{} `json:"normals"`
		// Normal          map[string]interface{} `json:"normal"`
		DirectToScatter float64 `json:"directToScatter"`
		Reflection      float64 `json:"reflection"`
		Roughness       float64 `json:"roughness"`
		Metallic        float64 `json:"metallic"`
		Index           int     `json:"index"`
		Specular        float64 `json:"specular"`
		ColorR          float64 `json:"colorR"`
		ColorG          float64 `json:"colorG"`
		ColorB          float64 `json:"colorB"`
		ColorA          float64 `json:"colorA"`
	}
- `POST /submitShader`: Submit shader configuration
    type ShaderParam struct {
	Type       string                 `json:"type"`
	Parameters map[string]interface{} `json:"params"`
}
- `GET /getCameraPosition`: Retrieve current camera position
type Position struct {
	X       float64 `json:"x"`
	Y       float64 `json:"y"`
	Z       float64 `json:"z"`
	CameraX float64 `json:"cameraX"`
	CameraY float64 `json:"cameraY"`
}
- `POST /moveToPosition`: Move camera to specified position
type Position struct {
	X       float64 `json:"x"`
	Y       float64 `json:"y"`
	Z       float64 `json:"z"`
	CameraX float64 `json:"cameraX"`
	CameraY float64 `json:"cameraY"`
}

### Backend Architecture
The backend is asynchronous and runs in a separate Go routine. It leverages Go's unsafe functionality to avoid state management complexities and mutex overhead, resulting in improved performance.

## Frontend Components

### Color Picker
- Color Channels (Multiplayer Enabled)
  - Red (R): 0-256
  - Green (G): 0-256
  - Blue (B): 0-256
  - Alpha (A): 0-256

#### Features
- Color Preview: Displays the currently selected color
- Triangle Color Application: Allows setting color for a selected triangle

### Texture Color Picker
#### Texture Selector
- Texture Range: 1-128
- Texture Preview: 128 × 128 × 4 float resolution

#### Normal Map
- Resolution: 128 × 128 × 3
- Normalization Range: -1 to 1
- Backend Conversion: Normalized to vector

##### Additional Features
- Upload Normal Map Button
- "No Normal" Button: Opens online normal map generator (https://cpetry.github.io/NormalMap-Online/)

#### Material Properties
- Reflection
- Direct to Scatter
- Roughness
- Metallic
- Specular

#### Texture Adjustment
- Sliders for material properties (0-1 range)
- Channel Multipliers (Red/Green/Blue/Alpha)

### Shader Menu
Purpose: Create post-processing shader chains (e.g., original image → contrast → tint → final image)

#### Shader Selection
- Shader Selector
- Add Shader Button
- Submit Shader Menu Button

#### Shader Parameters
Common Parameters:
- `amount`: Proportion of modified image to add to render
- `multipass`: Number of consecutive shader applications

Supported Shaders:
1. **Contrast**
   - Amount
   - Multipass
   - Contrast strength

2. **Tint**
   - Amount
   - Multipass
   - Tint color
   - Tint strength

3. **Bloom**
   - Amount
   - Multipass
   - Threshold
   - Intensity

4. **BloomV2**
   - Similar to Bloom with slight variations

5. **Sharpness**
   - Amount
   - Multipass
   - Filter strength

6. **Color Mapping**
   - Amount
   - Multipass
   - Color channels (R/G/B)

7. **Chromatic Aberration**
   - Amount
   - Multipass
   - Filter strength

8. **Edge Detection**
   - Uses Sobel filter
   - Amount
   - Multipass
   - Edge enhancement strength
   - Edge color (R/G/B)

9. **Lighten**
   - Amount
   - Multipass
   - Filter strength

### Render Options
#### Camera Management
- Submit Render Options
- Get Camera Position
- Hide/Show Camera Position
- Move Camera to Specific Position

#### Main Rendering Parameters
- **Depth**: Number of reflections to render
- **Scatter**: Number of rays scattered from surface (increases detail)

#### Lighting Parameters
- Light Intensity
- Light Color (R/G/B)
- Field of View
- Gamma (Contrast and brightness between dark and light tones)

#### Render Settings
- Snap Light to Camera
- Raymarching (Currently not implemented)
- Performance Mode
  - Removes `wg.Wait`
  - Potentially less smooth rendering
  - Maximizes hardware utilization

#### Render Modes
- Classic: Standard rendering
- Normal: Renders surface normals
- Distance: Currently not correctly implemented

### Volume Picker
#### Volume Color Management
- Submit Volume Colors
- Color Randomness
- Render Voxels Toggle
- Overwrite Voxels
- Add Randomness to Painting
- Convert Voxels to Smoke

#### Volume Properties
- Voxel Color Selector
- Smoke Color
- Density
- Transmittance (Volume transparency)


# Architektúra Projektu GO-Draaw

Projekt je rozdelený na dve hlavné časti:

## 1. Frontend
- Implementovaný v **Vue.js**
- Slúži ako používateľské rozhranie pre vizualizáciu a interakciu
- Poskytuje nástroje pre úpravu farieb, textúr, shaderov a renderovacích nastavení
- Zabezpečuje okamžitú spätnú väzbu na zmeny parametrov

## 2. Backend
- Vyvinutý v **Go (Golang)** s použitím **Echo frameworku**
- Pozostáva z dvoch hlavných komponentov:
  - **Ray-tracing engine**: Jadro výpočtového systému pre renderovanie
  - **Webový server**: Zabezpečuje komunikáciu s frontendovou časťou

Backend beží asynchrónne vo vlastných go-rutinách, čo minimalizuje potrebu zložitého manažmentu stavu a používania mutexov s pozitim usefe, čím sa dosahuje vyšší výkon a lepšia odozva systému.

Táto architektúra umožňuje efektívne oddelenie prezentačnej vrstvy od výpočtovej, pričom zachováva vysokú mieru interaktivity pre používateľa a zároveň poskytuje výkonný rendering komplexných 3D scén.

BVH a jej implementacia
    zo zaciatku som porovanaval ray itersection s trojuholnikom pre kazdy trojuholnik co je velmi neefektivny postup pri rastucom pocte trojuholnikov v dosledku toho som mal problemy s vykonom vramci riesenia tohto problemu som implementoval BoudingBoxi pre trojuuhoniky kedze porovanavanies zasahu s Boxom je jednuchsie no ani toto riesenie nedospelo k pozadovanim vysledkom vramci riesenia tochto problemu som implemetoval BVH ktora deli priesotor na mensie casti tento krok zvysil vykonost 

    BVHNode 
        type BVHNode struct { // size=136 (0x88)
            Left, Right *BVHNode
            BoundingBox [2]Vector
            Triangles   TriangleSimple
            active      bool
        }

        v pribehu developmentu sa BVH node neustale zvecsoval v dosledku pridavani novich parametrov ako su Normal Vector pre trojuholnik ci materialovych vlasnosti preto pre verziu V4 bola vytvorena nova BVHLean

        type BVHLeanNode struct { // size=72 (0x48)
            Left, Right  *BVHLeanNode
            TriangleBBOX TriangleBBOX
            active       bool
        }
        
        pre tuto verziu bolo upravene ze Trojuholnik aj Bounding Box je zluceny do jedneho structu 

        type TriangleBBOX struct { // size=52 (0x34)
            V1orBBoxMin, V2orBBoxMax, V3 Vector
            normal                       Vector
            id                           int32
        }

        taktiez pre trojuhonik boli odstranene materialove vlasnosti a boli presunute do texturi ktora a dana textura sa ziskava na zaklade ID

        taktiez vramci optimalizacie bolo experimentovane s array reprezentaciu kede plati ze lava node je n*2 a prava n*2+1
        type BVHArray struct { // size=65538508 (0x3e809cc)
            triangles [NumNodes]TriangleBBOX
            textures  [128]Texture
        }

        k danej verzi bolo implementovane aj testy a preukazalo sa ze dana verzie je klasickej implementacii moze byt vylepsena na zaklade mojich testov https://github.com/DarkBenky/testBinaryTree kde klasicka implementacia 278,146 ns/op a fix array reprezentacia 218,831 ns/op tento dosledok v dosledku lepsieho cashovania hodnot ked su hned za sebou zial k danej implementaciu som nestihol dokoncit vramci limitoveneho casu 






# BVH a jej implementácia

V procese optimalizácie ray-tracingu bola implementácia efektívnej akceleračnej štruktúry kľúčovým faktorom pre zlepšenie výkonu. Evolúcia riešenia prešla niekoľkými fázami:

## Vývojová cesta
1. **Naivný prístup** - Pôvodná implementácia testovala prienik lúča s každým trojuholníkom v scéne, čo viedlo k lineárnej časovej zložitosti O(n) a výrazne limitovalo výkon pri rastúcom počte trojuholníkov.

2. **Bounding Box optimalizácia** - Ako prvý krok optimalizácie boli implementované ohraničujúce boxy (Bounding Boxes) pre skupiny trojuholníkov, čo umožnilo rýchlejšie vylúčenie objektov mimo lúča. Toto zlepšenie však stále nebolo dostatočné pre komplexné scény.

3. **BVH implementácia** - Finálnym riešením bola implementácia Bounding Volume Hierarchy (BVH), ktorá hierarchicky organizuje priestor a umožňuje efektívne prechádzanie len relevantných častí scény, čím znižuje časovú zložitosť na približne O(log n).

## Evolúcia BVH štruktúry

### Pôvodná BVH implementácia
```go
type BVHNode struct { // veľkosť=136 (0x88) bajtov
    Left, Right *BVHNode
    BoundingBox [2]Vector
    Triangles   TriangleSimple
    active      bool
}
```

Prvá verzia BVH používala štandardnú stromovú štruktúru s ukazovateľmi na ľavý a pravý podstrom. Táto implementácia však trpela rastúcou veľkosťou uzlov kvôli pridávaniu materiálových vlastností a normálových vektorov pre trojuholníky.

### Optimalizovaná BVHLean
```go
type BVHLeanNode struct { // veľkosť=72 (0x48) bajtov
    Left, Right  *BVHLeanNode
    TriangleBBOX TriangleBBOX
    active       bool
}

type TriangleBBOX struct { // veľkosť=52 (0x34) bajtov
    V1orBBoxMin, V2orBBoxMax, V3 Vector
    normal                       Vector
    id                           int32
}
```

Pre verziu V4 bola vytvorená optimalizovaná implementácia BVHLean, ktorá:
- Zmenšila veľkosť uzla takmer na polovicu (zo 136 bajtov na 72 bajtov)
- Zlúčila ohraničujúci box a trojuholník do jednej štruktúry pre lepšiu lokalitu dát
- Odstránila priame ukladanie materiálových vlastností v uzle a nahradila ich systémom ID odkazov na textúry

### Experimentálna array-based implementácia
```go
type BVHArray struct { // veľkosť=65538508 (0x3e809cc) bajtov
    triangles [NumNodes]TriangleBBOX
    textures  [128]Texture
}
```

V rámci ďalšej optimalizácie bola experimentálne vytvorená array-based reprezentácia BVH, kde:
- Uzly sú uložené v súvislom poli miesto rozptýlených alokácií
- Vzťahy medzi uzlami sú implicitné (ľavý potomok má index 2n, pravý 2n+1)
- Zlepšuje sa lokalita referencií a efektivita cache pamäte procesora

Testovanie preukázalo 21% zlepšenie výkonu oproti klasickej implementácii (218,831 ns/op vs. 278,146 ns/op) vďaka lepšiemu cache využitiu pri sekvenčnom prístupe k dátam.

## Výkonnostné výsledky
Implementácia Array-based BVH poskytla merateľné zlepšenie výkonu:
- Klasická implementácia: 278,146 ns/op
- Array-based implementácia: 218,831 ns/op
- Zlepšenie: ~21%

Testovanie dostupné na: https://github.com/DarkBenky/testBinaryTree

Array-based implementácia zostala v experimentálnej fáze z dôvodu časových obmedzení projektu, ale predstavuje sľubný smer pre ďalší vývoj.



## TraceRay

The original function that provides basic ray tracing functionality:

- Uses BVH structure for intersection tests
- Performs basic scattered light calculation with a cosine-weighted hemisphere sampling
- Calculates direct reflections and specular highlights using a simple lighting model
- Uses a recursive approach for depth-based bounces
- Performs shadow calculation by casting shadow rays
- Combines direct lighting, scattered lighting, and reflections linearly

Mean Frame Time  Std Frame Time  Min Frame Time  Bottom Frame Time 10%  Top Frame Time 10%  Max Frame Time  Median Frame Time
45535.5483      82965.2246           626.0                  802.4             86894.6       1084617.0            40969.0


An improved version that:

- Organizes the code more logically, separating direct illumination, indirect illumination, and reflection
- Adds roughness-based perturbation to reflection directions
- Implements more physically-based energy conservation
- Uses hemisphere sampling with improved scattering logic
- Combines components using a more physically accurate approach
- Better handles the energy balance between diffuse and specular

## TraceRayV3

A PBR (Physically Based Rendering) approach that:

- Implements Fresnel-Schlick approximation for reflection calculation
- Uses GGX distribution for microfacet-based specular highlights
- Calculates important dot products (NdotL, NdotV, NdotH) for PBR calculations
- Better simulates material properties like metalness and roughness
- Employs more accurate energy conservation for combining components
- Returns a single color value

Mean Frame Time  Std Frame Time  Min Frame Time  Bottom Frame Time 10%  Top Frame Time 10%  Max Frame Time  Median Frame Time
51327.2233      98163.7160           627.0                  782.0             94945.9       1108286.0            44979.0

## TraceRayV3Advance

An extension of TraceRayV3 that:
- Returns additional data: color, distance, and normal vector
- Allows for more advanced post-processing techniques
- Otherwise uses the same PBR approach as TraceRayV3
- Supports storing data for deferred shading techniques

Mean Frame Time  Std Frame Time  Min Frame Time  Bottom Frame Time 10%  Top Frame Time 10%  Max Frame Time  Median Frame Time
 54768.3617     104101.6718          1925.0                 2221.7             99104.7       1109015.0            47394.5

## TraceRayV3AdvanceTexture

A texture-enabled version that:
- Integrates material properties from textures
- Takes a textureMap parameter to access texture data
- Returns color and normal information for post-processing
- Uses specialized BVH traversal (`IntersectBVH_Texture`) for texture support
- Applies texture data to material parameters like roughness and metallic

Mean Frame Time  Std Frame Time  Min Frame Time  Bottom Frame Time 10%  Top Frame Time 10%  Max Frame Time  Median Frame Time
55282.6100     105146.7734          1194.0                 1324.9             99664.7       1113191.0            47701.0

## TraceRayV4AdvanceTexture

An optimized texture-enabled version that:
- Uses the lightweight `BVHLeanNode` structure instead of the standard BVH
- Employs an optimized intersection function (`IntersectBVHLean_Texture`)
- Otherwise similar to TraceRayV3AdvanceTexture in functionality
- Returns color and normal information

Mean Frame Time  Std Frame Time  Min Frame Time  Bottom Frame Time 10%  Top Frame Time 10%  Max Frame Time  Median Frame Time
49291.8733      96904.0536          1772.0                 2085.8             86554.2       1101608.0            40759.0

## TraceRayV4AdvanceTextureLean

The most optimized version that:
- Only returns color information (no normal vectors or distance)
- Uses the minimal `IntersectBVHLean_TextureLean` intersection
- Reduces memory usage and minimizes data structure overhead
- Maintains all PBR calculations but simplifies the return structure
- Specifically designed for pure color rendering without other data

Mean Frame Time  Std Frame Time  Min Frame Time  Bottom Frame Time 10%  Top Frame Time 10%  Max Frame Time  Median Frame Time
46876.2583      95782.8693          1570.0                 2005.9             84855.7       1095377.0            40367.5

## Key Evolution Points:

1. **Rendering Model**: From a basic model (TraceRay) to a full PBR model (V3 and beyond)
2. **Data Return**: From just color to color+distance+normal to just color again for optimization
3. **BVH Usage**: From standard BVH to optimized lean BVH structures
4. **Material Simulation**: From basic reflection to full PBR with metalness, roughness, and Fresnel
5. **Texture Support**: Added in V3AdvanceTexture and maintained in V4 variations
6. **Memory Usage**: Progressively optimized, especially in the V4Lean variant
7. **Performance**: Each version making trade-offs between features and speed

These functions represent a typical evolution path in ray tracer development, moving from correctness to performance optimization while maintaining physically based rendering principles.


# Understanding FresnelSchlick and GGXDistribution Functions

These two functions are key components of physically-based rendering (PBR), which aims to simulate how light interacts with surfaces in a realistic way.

## FresnelSchlick Function

```go
func FresnelSchlick(cosTheta, F0 float32) float32 {
    return F0 + (1.0-F0)*math32.Pow(1.0-cosTheta, 5)
}
```

### Purpose
The FresnelSchlick function approximates the **Fresnel effect**, which describes how the amount of light reflected vs. refracted changes based on the viewing angle.

### Parameters
- `cosTheta`: The cosine of the angle between the view direction and the surface normal
- `F0`: Base reflectivity of the material at normal incidence (looking straight at the surface)

### How It Works
1. When viewing a surface straight on (`cosTheta` near 1), the reflection is close to the material's base reflectivity (`F0`)
2. When viewing at grazing angles (`cosTheta` near 0), almost all light is reflected regardless of material type
3. The function uses Schlick's approximation, which is computationally efficient while providing good visual results

### Practical Effects
- For metals (conductors), `F0` is typically high (0.5-1.0), resulting in strong reflections
- For non-metals (dielectrics), `F0` is typically low (0.02-0.05), with reflections mainly visible at grazing angles
- This creates the effect where water, glass, or plastic surfaces become mirror-like when viewed at shallow angles

## GGXDistribution Function

```go
func GGXDistribution(NdotH, roughness float32) float32 {
    alpha := roughness * roughness
    alpha2 := alpha * alpha
    NdotH2 := NdotH * NdotH
    denom := NdotH2*(alpha2-1.0) + 1.0
    return alpha2 / (math32.Pi * denom * denom)
}
```

### Purpose
The GGXDistribution function models the **microfacet distribution** of a surface, describing how microscopic surface irregularities affect light reflection.

### Parameters
- `NdotH`: Dot product between the surface normal and the half vector (the vector halfway between view and light directions)
- `roughness`: Surface roughness parameter (0 = perfectly smooth, 1 = very rough)

### How It Works
1. The function implements the GGX/Trowbridge-Reitz distribution, which is considered one of the most realistic microfacet distributions
2. The `alpha` parameter is derived from roughness (squared to match artistic expectations)
3. The distribution describes the statistical probability that microfacets are oriented in the half-vector direction
4. For smooth surfaces (low roughness), the distribution creates a tight, intense specular highlight
5. For rough surfaces (high roughness), the distribution spreads reflection over a wider area, creating a more diffuse appearance

### Practical Effects
- Controls the size and intensity of specular highlights
- Smooth surfaces (low roughness) have small, bright highlights
- Rough surfaces (high roughness) have large, dim highlights
- Properly handles the "bright edge" phenomenon seen on curved objects

Together, these two functions form the core of the specular BRDF (Bidirectional Reflectance Distribution Function) in your PBR renderer, accurately modeling how different materials reflect light based on their physical properties.

# Voxel and Volume Rendering

The GO-Draaw project implements two related but distinct rendering techniques: voxel rendering and volume rendering. Both utilize a unified data structure but employ different rendering approaches to achieve their visual effects.

## Core Data Structures

```go
type Block struct {
    Position   Vector
    LightColor ColorFloat32  // Used for solid voxel rendering
    SmokeColor ColorFloat32  // Used for volumetric effects
}

type VoxelGrid struct {
    BlocksPointer  unsafe.Pointer   // Direct memory access for performance
    Blocks         []Block          // Slice view of blocks
    BBMin          Vector           // Bounding box minimum coordinates
    BBMax          Vector           // Bounding box maximum coordinates
    Resolution     int              // Grid resolution
    VolumeMaterial VolumeMaterial   // Material properties for volumetric rendering
}
```

The implementation leverages Go's `unsafe` package to achieve more efficient memory access patterns and avoid bounds checking during ray traversal, resulting in significant performance improvements.

## Voxel Rendering Implementation

Voxel rendering treats each voxel as a discrete, solid element with defined boundaries. The implementation uses a ray-marching technique through the grid, checking for occupied voxels along the ray path.

```go
func (v *VoxelGrid) IntersectVoxel(ray Ray, steps int, light Light) (ColorFloat32, bool) {
    // Find entry and exit points of the ray with the bounding box
    hit, entry, exit := BoundingBoxCollisionEntryExitPoint(v.BBMax, v.BBMin, ray)
    if !hit {
        return ColorFloat32{}, false  // Ray doesn't intersect the grid
    }

    // Calculate step size based on total distance and desired steps
    stepSize := exit.Sub(entry).Mul(1.0 / float32(steps))
    
    // March along the ray
    currentPos := entry
    for i := 0; i < steps; i++ {
        // Check for voxel at current position using unsafe direct access
        block, exists := v.GetVoxelUnsafe(currentPos)
        if exists {
            // Shadow calculation
            lightStep := light.Position.Sub(currentPos).Mul(1.0 / float32(steps*2))
            lightPos := currentPos.Add(lightStep)
            
            // Cast shadow ray toward light source
            for j := 0; j < steps; j++ {
                _, shadowHit := v.GetVoxelUnsafe(lightPos)
                if shadowHit {
                    return block.LightColor.MulScalar(0.05), true  // Point in shadow
                }
                lightPos = lightPos.Add(lightStep)
            }
            
            // Calculate light attenuation based on distance
            lightDistance := light.Position.Sub(currentPos).Length()
            attenuation := ExpDecay(lightDistance)
            blockColor := block.LightColor.MulScalar(attenuation)
            
            return blockColor, true  // Visible voxel with lighting
        }
        currentPos = currentPos.Add(stepSize)
    }
    
    return ColorFloat32{}, false  // No intersection found
}
```

### Key Features:
- Binary visibility (voxel exists or doesn't)
- Hard shadow calculation
- Exponential light decay with distance
- Simple direct lighting model

## Volume Rendering Implementation

Volume rendering treats the grid as a continuous medium with varying densities. It implements physically-based light scattering and absorption through participating media.

```go
func (v *VoxelGrid) Intersect(ray Ray, steps int, light Light, volumeMaterial VolumeMaterial) ColorFloat32 {
    hit, entry, exit := BoundingBoxCollisionEntryExitPoint(v.BBMax, v.BBMin, ray)
    if !hit {
        return ColorFloat32{}
    }

    // Physical parameters for light interaction
    const (
        extinctionCoeff  = 0.5          // Controls light absorption
        scatteringAlbedo = 0.9          // Ratio of scattering to absorption
        asymmetryParam   = float32(0.3) // Controls scattering direction bias
    )

    stepSize := exit.Sub(entry).Mul(1.0 / float32(steps))
    stepLength := stepSize.Length()

    var accumColor ColorFloat32
    transmittance := volumeMaterial.transmittance  // Initial transparency

    currentPos := entry
    for i := 0; i < steps; i++ {
        block, exists := v.GetBlockUnsafe(currentPos)
        if !exists {
            currentPos = currentPos.Add(stepSize)
            continue
        }

        density := volumeMaterial.density
        extinction := density * extinctionCoeff

        // Henyey-Greenstein phase function calculation
        lightDir := light.Position.Sub(currentPos).Normalize()
        cosTheta := ray.direction.Dot(lightDir)
        g := asymmetryParam
        phaseFunction := (1.0 - g*g) / (4.0 * math32.Pi * math32.Pow(1.0+g*g-2.0*g*cosTheta, 1.5))

        // Calculate light attenuation through the volume
        lightRay := Ray{origin: currentPos, direction: lightDir}
        lightTransmittance := v.calculateLightTransmittance(lightRay, light, density)

        // Calculate scattered light contribution
        scattering := extinction * scatteringAlbedo * phaseFunction * 2.0

        // Apply Beer-Lambert law for light absorption
        sampleExtinction := math32.Exp(-extinction * stepLength)
        transmittance *= sampleExtinction

        // Accumulate color with proper physical weighting
        lightContribution := ColorFloat32{
            R: block.SmokeColor.R * light.Color[0] * lightTransmittance * scattering,
            G: block.SmokeColor.G * light.Color[1] * lightTransmittance * scattering,
            B: block.SmokeColor.B * light.Color[2] * lightTransmittance * scattering,
            A: block.SmokeColor.A * density,
        }

        // Add contribution to final color, weighted by current transmittance
        accumColor = accumColor.Add(lightContribution.MulScalar(transmittance))

        // Early exit optimization when transmittance becomes negligible
        if transmittance < 0.001 {
            break
        }

        currentPos = currentPos.Add(stepSize)
    }

    // Ensure proper alpha channel normalization
    accumColor.A = math32.Min(accumColor.A, 1.0)
    return accumColor
}
```

### Key Features:
- Physically-based light scattering using Henyey-Greenstein phase function
- Beer-Lambert law for light absorption
- Progressive light accumulation with proper transmittance
- Support for variable density throughout the volume
- Early termination optimization for rays with negligible remaining transmittance

## Performance Optimizations

1. **Unsafe Memory Access**: The implementation uses `unsafe.Pointer` for direct memory access to the voxel grid, bypassing Go's bounds checking for improved performance.

2. **Early Ray Termination**: The volume renderer stops ray marching when transmittance falls below a threshold (0.001), avoiding unnecessary calculations.

3. **Bounding Box Pre-Testing**: Both renderers first test ray intersection with the grid's bounding box before performing detailed traversal.

4. **Distance-Based Light Attenuation**: Light contribution is attenuated based on distance, providing realistic falloff without expensive calculations.

## Interactive Editing Features

The system supports several interactive editing operations:

- **Adding/Removing Voxels**: Users can interactively add or remove voxels from the grid.
- **Color Manipulation**: Voxel colors can be changed individually or in groups.
- **Conversion to Volumes**: Solid voxels can be converted to volumetric data for smoke/fog effects.
- **Material Parameter Adjustment**: Density and transmittance can be tuned for different visual effects.

## Potential Future Improvements

1. **Multi-Resolution Voxel Grid**: Implement an octree or sparse voxel octree (SVO) structure to efficiently handle varying detail levels.

2. **Multiple Scattering Simulation**: Extend the volume renderer to simulate multiple light bounces within the volume for more realistic effects.

3. **GPU Acceleration**: Port the voxel and volume traversal algorithms to compute shaders for parallel processing.

4. **Procedural Volume Generation**: Add support for procedural noise functions to generate natural-looking volumes.

5. **Adaptive Sampling**: Implement adaptive step sizes based on density gradients to improve detail in areas of high complexity.

6. **Voxel Instancing**: Allow repeated instances of the same voxel structure with different transformations, useful for complex scenes.

7. **Vectorized Calculations**: Utilize SIMD instructions through Go's assembly options for faster numerical calculations.

## Volume Physics Models

The current implementation includes a simplified physical model based on:

- **Beer-Lambert Law**: For light absorption through participating media
- **Henyey-Greenstein Phase Function**: For anisotropic light scattering
- **Exponential Decay**: For light attenuation with distance

These physical models provide a foundation for realistic volumetric effects like fog, smoke, and clouds, which can be further enhanced through parameter tuning and additional physical simulations.

# Raymarching Implementation

The current raymarching implementation is limited to spheres due to their simple distance function:

```go
func Distance(v1, v2 Vector, radius float32) float32 {
    // Use vector subtraction and dot product instead of individual calculations
    diff := v1.Sub(v2)
    return diff.Length() - radius
}
```

## Current Status
- Only supports sphere primitives
- Uses BVH for acceleration
- Basic implementation without advanced features

## Future Development Plans

### Expanded Primitive Support
To enhance the raymarching capabilities, I plan to implement additional geometric primitives:

```go
// Box SDF
func BoxSDF(point, boxCenter, boxDimensions Vector) float32 {
    localPoint := point.Sub(boxCenter)
    q := Vector{
        math32.Abs(localPoint.X) - boxDimensions.X/2,
        math32.Abs(localPoint.Y) - boxDimensions.Y/2,
        math32.Abs(localPoint.Z) - boxDimensions.Z/2,
    }
    
    return math32.Min(math32.Max(q.X, math32.Max(q.Y, q.Z)), 0.0) + 
           Vector{math32.Max(q.X, 0), math32.Max(q.Y, 0), math32.Max(q.Z, 0)}.Length()
}

// Torus SDF
func TorusSDF(point, center Vector, majorRadius, minorRadius float32) float32 {
    localPoint := point.Sub(center)
    q := Vector{Vector{localPoint.X, 0, localPoint.Z}.Length() - majorRadius, localPoint.Y, 0}
    return q.Length() - minorRadius
}

// Cylinder SDF
func CylinderSDF(point, center Vector, height, radius float32) float32 {
    localPoint := point.Sub(center)
    d := Vector{Vector{localPoint.X, 0, localPoint.Z}.Length() - radius, math32.Abs(localPoint.Y) - height/2, 0}
    return math32.Min(math32.Max(d.X, d.Y), 0) + 
           Vector{math32.Max(d.X, 0), math32.Max(d.Y, 0), 0}.Length()
}
```

### SDF Operations
To enable complex object creation through operations like union, intersection, and difference:

```go
// Union of two SDFs
func SdfUnion(d1, d2 float32) float32 {
    return math32.Min(d1, d2)
}

// Smooth union with blending
func SdfSmoothUnion(d1, d2, k float32) float32 {
    h := math32.Max(k-math32.Abs(d1-d2), 0.0)
    return math32.Min(d1, d2) - h*h*0.25/k
}

// Intersection of two SDFs
func SdfIntersection(d1, d2 float32) float32 {
    return math32.Max(d1, d2)
}

// Subtraction of SDF2 from SDF1
func SdfDifference(d1, d2 float32) float32 {
    return math32.Max(d1, -d2)
}
```

### Implementation Roadmap

1. **Primitive Expansion**
   - Implement basic primitives (box, torus, cylinder)
   - Add parameter controls in the UI for each primitive type

2. **SDF Operations**
   - Implement Boolean operations (union, intersection, difference)
   - Add smooth blending between shapes for organic forms

3. **Performance Optimization**
   - Extend BVH acceleration structure to work with all SDF primitives
   - Implement spatial partitioning specific to raymarching

4. **User Interface**
   - Create a dedicated raymarching control panel
   - Add visual feedback for SDF operations

5. **Advanced Features**
   - Domain repetition for creating patterns
   - Noise-based deformations for organic shapes
   - Material assignment for SDF objects

This expanded raymarching system will enable the creation of complex shapes through constructive solid geometry operations, allowing users to build intricate models that would be difficult to achieve with traditional triangle-based geometry.
