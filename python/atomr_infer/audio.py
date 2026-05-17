"""``atomr_infer.audio`` — facade over ``atomr_infer._native.audio``.

Exposes the audio-modality data types shared by speech-to-text (STT),
text-to-speech (TTS), and Audio2Face (A2F) runtimes:

- ``AudioFormat`` — PCM / Opus / MP3 / FLAC / WAV format tags
- ``AudioParams`` — sample rate / channels / format tuple
- ``AudioPayload`` — serializable audio source (bytes, path, URL)
- ``VoiceRef`` — TTS voice selector (named, ID, cloned)
- ``TranscribeOptions`` / ``SynthOptions`` / ``A2FOptions`` — per-modality knobs
- ``SpeechBatch`` — TTS request batch
- ``AudioBatch`` — STT *and* A2F request batch (modality selected via options)
- ``TranscriptChunk`` / ``SpeechChunk`` / ``BlendshapeChunk`` — per-modality output chunks
- ``WordTiming`` / ``Viseme`` / ``AlignmentDelta`` — alignment primitives

Example — text-to-speech batch::

    from atomr_infer.audio import SpeechBatch, SynthOptions, VoiceRef

    batch = SpeechBatch(
        request_id="r1",
        model="tts-1",
        text="hello world",
        voice=VoiceRef.named("alloy"),
        options=SynthOptions(),
        stream=True,
        emit_alignment=False,
    )

Example — speech-to-text batch::

    from atomr_infer.audio import (
        AudioBatch, AudioFormat, AudioParams, AudioPayload, TranscribeOptions,
    )

    payload = AudioPayload.from_path("hello.wav", AudioParams(16000, 1, AudioFormat.WAV))
    batch = AudioBatch.transcribe(
        request_id="r1", model="whisper-1", input=payload,
        options=TranscribeOptions(), stream=False,
    )

Example — Audio2Face batch::

    from atomr_infer.audio import (
        A2FOptions, AudioBatch, AudioFormat, AudioParams, AudioPayload,
    )

    payload = AudioPayload.from_path("speech.wav", AudioParams(16000, 1, AudioFormat.PCM16_LE))
    batch = AudioBatch.audio2face(
        request_id="r1", model="audio2face-3d", input=payload,
        options=A2FOptions(), stream=True,
    )
"""

from ._native import audio as _sub

globals().update({k: getattr(_sub, k) for k in dir(_sub) if not k.startswith("_")})
__all__ = [k for k in dir(_sub) if not k.startswith("_")]
