These upstream license texts fill omissions in the published Cargo archives.
The build script matches exact crate versions; new versions need a fresh review.

- `flatbuffers-25.12.19.txt`: [Google FlatBuffers LICENSE](https://github.com/google/flatbuffers/blob/7e163021e59cca4f8e1e35a7c828b5c6b7915953/LICENSE), at the commit recorded in the crate's `.cargo_vcs_info.json`.
- `tracelogging-1.2.3.txt`: [Microsoft TraceLogging LICENSE](https://github.com/microsoft/tracelogging/blob/4d3c8b02da568d34daf977db7052a819ab4b18ec/LICENSE), at the commit recorded in both `tracelogging` and `tracelogging_macros` version 1.2.3.

All other dependency notices come directly from Cargo's resolved source checkout.
