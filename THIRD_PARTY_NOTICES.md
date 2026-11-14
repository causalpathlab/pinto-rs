# Third-party notices

pinto-rs is distributed under the MIT License (see `LICENSE`). Two of its
spatial baselines are ports of code from other MIT-licensed projects. Each
section below names the project, where it is, the commit the port follows
and the module that holds the port, and reproduces that project's license
as its LICENSE file states it. The papers behind the methods are cited in
each module's documentation.

## SpatialDE

- Project: SpatialDE (SpatialDE2's tissue segmentation)
- URL: https://github.com/PMBio/SpatialDE
- Commit followed: 6dbdb559a566d639da0f6fd3acc9e17104c0674f
- Ported in: `src/spatial_baselines/spatialde2.rs` (`pinto spatialde2`)

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
- Ported in: `src/spatial_baselines/stagate.rs` (`pinto stagate`)

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
