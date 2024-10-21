
# TODO List

- [ ] Implement SIMD instructions
- [X] Merge the latest branch with main
- [X] Implement bloom and other post-processing effects
- [ ] Implement Shaders for Color Grating, Contract etc.
- [ ] Optimize for speed
- [ ] Create flame graphs
- [X] Trace rays to corners and draw triangles:
  - Create a box/cone representing the camera's viewing area
  - Find all triangles inside it
  - Order them based on distance
  - Draw them (consider adding shading based on normals)
- [ ] Fix ray tracer:
  - Fix uniforms ([link](https://github.com/tinne26/kage-desk/blob/main/docs/tutorials/intro/06_uniforms.md))
    - Note: Uniforms start with capital letters
- [X] Implement specular lighting
- [X] Add Coloring of Triangles
- [X] Fix camera movement(It work not greatly)
  - [X] Add the previous camera Projection - Done but i don't like it maybe look again in to it
  - <https://chatgpt.com/c/6709b2d9-4770-8004-a46b-59e572c0b0cc>
- [X] Optimize  PrecomputeScreenSpaceCoordinatesSphere ( go routines, if no movement no change)
