# atomr-infer-runtime-assemblyai

AssemblyAI speech-to-text runtime for [`atomr-infer`](https://github.com/rustakka/atomr-infer).

Implements `AudioRunner` against `wss://streaming.assemblyai.com/v3/ws`
(AssemblyAI's Universal-Streaming v3 protocol) on top of the shared
`atomr-infer-runtime-ws-core` transport.

## Build profiles

| Build                                                                       | Result                                                |
|-----------------------------------------------------------------------------|-------------------------------------------------------|
| `cargo build -p atomr-infer-runtime-assemblyai`                             | Stub — `execute_audio` returns `Internal("stt-assemblyai feature disabled at build time")`. |
| `cargo build -p atomr-infer-runtime-assemblyai --features stt-assemblyai`   | Real path — WSS streaming + partial → final-per-turn progression. |

See `FR-STT-001` and `docs/audio-modalities.md` for the program-level context.
