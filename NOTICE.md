# Sources and distribution scope

This repository documents an experimental engineering model. It does not claim biological fidelity, general scientific reviewing ability, or superiority of biological wiring.

- MaleCNS connectome: https://github.com/natverse/malecns and https://male-cns.janelia.org/ . Credit the original connectome researchers and consult their current data/citation terms.
- Referenced demonstration: https://huggingface.co/spaces/VIDraft/fruitfly-brain .
- Upstream graph-bearing artifact: https://huggingface.co/ngxson/fly-llm-hf at revision `65c677b3d566a2e9793d5f72999cdb441c6c0a9f`. SHA-256 `355f06c44d14e38af50e9c801f51839c37a0a56ac4ca016da2be9b3ae6f215ad`. The upstream artifact is downloaded separately; this project's model reads only its graph tensors and input-neuron indices, not its language parameters. Upstream terms remain applicable.
- Training-source XML: https://github.com/elifesciences/elife-article-xml at recorded tree revision `0c5058c62b6b4985c9fa0ac3c773ba8107350a2e`. Article-level license metadata was used in filtering. The original articles/XML are not redistributed here. Refer to each article's authors and license for reuse.
- Example source: Berg et al., *Sexual dimorphism in the complete Drosophila male central nervous system connectome*, https://doi.org/10.1016/j.cell.2026.08.015 . The example outputs contain short verbatim excerpts from this user-supplied paper. The PDF and full extracted text are excluded. Input PDF SHA-256: `1917e8c7bd79fd1492c6ecc071fc12fea4cf9b147e6f5f2981016c925fac846a`.

No project-wide reuse license has been selected. Public availability does not itself grant a new license to code, weights, training-derived assets, or third-party text. Third-party rights and terms are not superseded by this repository.
