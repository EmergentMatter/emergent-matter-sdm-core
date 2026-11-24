Keep open sweep cap depths continuous across segment switches in JAX and GLSL, including subdivided straight paths and cap faces partly covered by another leg.

Open sweeps now evaluate the profile at both endpoint caps in addition to the selected segment, in JAX and emitted GLSL. Shader consumers such as sdm-view should account for these additional profile evaluations, particularly with complex profiles. This is not a measured whole-sweep slowdown. Cap correction does not make general sweeps exact distance fields.
