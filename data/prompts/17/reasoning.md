# V17 one-dimensional rubric rationale

## Three dimensions

1. Visual evidence difficulty: How much usable visual evidence is available for the relevant objects. Combines image acquisition quality and target visibility: blur, noise, compression, exposure, illumination, and contrast; object size and available visible detail; occlusion and truncation.
2. Scene and instance complexity: Represents the amount and organization of visual content: object number, density, and diversity; clutter and distractors; crowding and object-object overlap; spatial separation and instance-boundary ambiguity.
3. Object interpretation demand: How difficult it is to interpret the object: pose, orientation, viewpoint, and foreshortening; deformation and atypical appearance; dependence on surrounding context; ambiguity between plausible object interpretations; category similarity when the task is class-aware detection.

*For class-agnostic localization, category distinctions are excluded. 

For assigning the overall score:
- overall Level 1 when all dimensions are Level 1
- overall Level 2 when the highest dimension is Level 2 and only one dimension reaches it
- overall Level 3 for one moderate dimension or at least two mild dimensions
- overall Level 4 for one severe dimension or at least two moderate dimension
- overall Level 5 for one extreme dimension or at least two severe dimensions



# Next?

Suggested six dimensions, but seems too much.
Dimension	Conditions already present in v16	What it measures
1. Signal quality	Blur, noise, resolution, lighting, shadows	How clearly the image records the visual evidence
2. Target scale and detail	Large/small objects, close-ups, sufficient resolution	Whether each relevant object has enough visible pixels
3. Scene load	Number of objects, complex scenes, background clutter	How much visual information must be processed
4. Spatial separability	Occlusion, overlap, truncation, unclear boundaries	How easily individual object instances can be separated and bounded
5. Geometric variation	Unusual pose, orientation, perspective, deformation	How far visible object geometry departs from a canonical presentation
6. Semantic ambiguity	Similar categories, atypical objects, context dependence, specialized objects	How difficult it is to determine what an object is