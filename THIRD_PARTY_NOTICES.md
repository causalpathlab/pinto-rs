# Third-party notices

pinto-rs is distributed under the MIT License (see `LICENSE`). Its
spatial baselines, and the single-cell pipeline of its simulation
baselines, follow code from other projects under the MIT and BSD 3-Clause
licenses. Each section below names the project, where it is, the commit
the port follows and the module that holds the port, and reproduces that
project's license as its LICENSE file states it. The papers behind the
methods are cited in each module's documentation.

## SpatialDE

- Project: SpatialDE (SpatialDE2's tissue segmentation)
- URL: https://github.com/PMBio/SpatialDE
- Commit followed: 6dbdb559a566d639da0f6fd3acc9e17104c0674f
- Ported in: `src/baselines/spatial/spatialde2.rs` (`pinto spatialde2`)

```text
MIT License

Copyright (c) 2020 Ilia Kats <i.kats@dkfz-heidelberg.de>, Stegle Group
Copyright (c) 2018 Teichmann Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## STAGATE_pyG

- Project: STAGATE_pyG (the PyTorch Geometric build of STAGATE)
- URL: https://github.com/QIFEIDKN/STAGATE_pyG (now served from
  https://github.com/RucDongLab/STAGATE_pyG)
- Commit followed: ae1158ca8cf1eb6bb8ee198298552d44c9ac21db
- Ported in: `src/baselines/spatial/stagate.rs` (`pinto stagate`)

```text
MIT License

Copyright (c) 2021 Kangning Dong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## STAGATE

- Project: STAGATE (the original TensorFlow 1 build, by the STAGATE
  authors)
- URL: https://github.com/zhanglabtools/STAGATE
- Commit followed: d6dd9f4747ff4f0356347ee9f662d9404bd68143
- Ported in: `src/baselines/spatial/stagate.rs` (`pinto stagate`): the
  steps that follow its tutorials and trainer where they and STAGATE_pyG
  differ (the fixed-K tutorial's preprocessing: no gene filter by default,
  the spatial network of `utils.py:Cal_Spatial_Net` with its radius
  option, and the sigmoid attention of its `model.py`)

```text
MIT License

Copyright (c) 2021 Kangning Dong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## scanpy

- Project: scanpy
- URL: https://github.com/scverse/scanpy
- Commit followed: 3962176e93ba148bc0f61443c517877d736f2b3b
- Ported in: `src/baselines/spatial/stagate.rs` (`pinto stagate`):
  `seurat_v3_hvg`, after `_highly_variable_genes_seurat_v3`
  (`highly_variable_genes(flavor="seurat_v3")`), and `filter_genes`, after
  `pp.filter_genes(min_cells)`; `src/baselines/single_cell.rs`
  (`pinto srt-sim-baseline`): `normalise_log` (`pp.normalize_total` to
  1e4 then `pp.log1p`, also stagate's normalisation),
  `gene_mean_dispersion`, `normalised_dispersion` and `hvg`
  (`highly_variable_genes(flavor="seurat")`), `scale` (`pp.scale` with
  `max_value=10`), and the recipe of `scanpy_pcs` and `cluster`
  (`pp.neighbors` then `tl.leiden`)

```text
BSD 3-Clause License

Copyright (c) 2025 scverse®
Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
