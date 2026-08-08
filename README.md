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

      A I O N
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
```

#### LICENSE
 This project is dual-licensed under the terms of the Eclipse Public License v2.0 (EPL-2.0) or (at your option) the GNU General Public License v2.0 or later (GPL-2.0-or-later). See [LICENSE](LICENSE) for details. The bindings under the `bindings` directory are licensed under the Apache License 2.0. See [bindings/LICENSE](bindings/LICENSE) for details.