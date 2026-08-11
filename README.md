```
 ._____A_____,
 |`          :\
 | `         : B
 |  `        :  \
 C   +-----A-----+
 |   :       :   :
 |__ : _A____:   :
 '   :        \  :
  `  :         B :
   ` :          \:
    `<_____A_____>

      A I O N (v0.0.1)
```

The tensor manipulation library to rule them all.

I think examples are most intuitive in how to learn to use a library. As such, look at [examples](examples) for zig core library usage and [bindings](bindings) for usage in downstream wrapper libraries. Currently, the bindings are used only for [Python](bindings/python) but its [examples](bindings/python/examples/) are more illustrative of the capabilities of the library in terms of use as a runtime. Similarly, you can see how to convert and build out models through the [scripts](scripts) directory (i.e take a look at the [Silero VAD](scripts/convert_silero_vad_to_aion.py) for a simple example of how to convert a model to the .aion format).

#### Features
- Graph IR based tensor computation/execution engine with a focus on high (in reality "pretty good") and portable performance.
- Singular .aion file for model weights, graph, and metadata to run directly without specifying a model graph or weights separately.
- Inference only (at the moment), no autograd.
- CPU: AOT compiled kernels for x86_64 and aarch64 (arm64) for pretty much top-notch inference performance.
- GPU: wgpu backend with pretty good cross-platform performance (70% of CUDA on RTX 4080).

#### Python
You can install the library through pip with the following commands:

```bash
pip install aion-engine
pip install aion-engine[gpu]
```

#### AION FILE FORMAT
This is a moving target at this point and I don't want to lock in bad decisions on graph IR and metadata. As such converted models to aion format right now require a versions of the library which support the same version as it. When I feel satisfied on where it is, backwards compatability will be added to the library and the file format will be locked in. This mostly concerns destructive changes, as additive changes (such as new graph ops) won't push the version of the format. The current version of the aion file format is v10.

#### LICENSE
 This project is dual-licensed under the terms of the Eclipse Public License v2.0 (EPL-2.0) or (at your option) the GNU General Public License v2.0 or later (GPL-2.0-or-later). See [LICENSE](LICENSE) for details. The bindings under the `bindings` directory are licensed under the Apache License 2.0. See [bindings/LICENSE](bindings/LICENSE) for details.