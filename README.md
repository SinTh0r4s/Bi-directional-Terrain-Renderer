# Terrain renderer for huge heightmaps for Claire

### TODO

- [ ] centralize lighting config for skybox & terrain
- [ ] ingrate frustrum culling in lod selector
- [ ] tune lod selection formula
- [ ] AABB don't line up with rendered terrain!?
  - probably fixes lighting calc & frustrum culling
  - [ ] does it remove gaps between chunks?
- [ ] double check that the correct neighbor LOD level are provided and no axes are switched/flipped
- fix skybox to stop sun from moving when getting closer to the corners

- [ ] add image overlay + ability to store camera settings when found
- [ ] add reverse rendering
- [ ] add export of reverse engineering + used image paths + camera info

- [ ] integrate in QGIS
