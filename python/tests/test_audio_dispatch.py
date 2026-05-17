"""Audio modality dispatch — Python facade smoke tests.

Exercises the PyO3-bound `audio` submodule for all four modalities:
- TTS via ``SpeechBatch``
- STT via ``AudioBatch.transcribe``
- Audio2Face via ``AudioBatch.audio2face``

Construction-level only — runner dispatch lives in the Rust integration
tests under ``crates/inference-testkit/tests/``. These tests confirm the
Python facade exposes the same types under the same names and that
attribute round-trips work, satisfying the M12 Python-parity exit
criterion.
"""

from atomr_infer.audio import (
    A2FOptions,
    AudioBatch,
    AudioFormat,
    AudioParams,
    AudioPayload,
    SpeechBatch,
    SynthOptions,
    TranscribeOptions,
    VoiceRef,
)


def test_audio_format_classattrs_exposed() -> None:
    """All seven AudioFormat variants reachable as class attributes."""
    for variant in (
        AudioFormat.PCM16_LE,
        AudioFormat.PCM24_LE,
        AudioFormat.PCM_F32_LE,
        AudioFormat.OGG_OPUS,
        AudioFormat.MP3,
        AudioFormat.FLAC,
        AudioFormat.WAV,
    ):
        assert variant is not None


def test_audio_params_round_trip() -> None:
    p = AudioParams(16000, 1, AudioFormat.PCM16_LE)
    assert p.sample_rate_hz == 16000
    assert p.channels == 1


def test_audio_payload_from_bytes() -> None:
    params = AudioParams(16000, 1, AudioFormat.PCM16_LE)
    payload = AudioPayload.from_bytes(b"\x00\x00\x00\x00", params)
    assert payload is not None


def test_audio_payload_from_path() -> None:
    params = AudioParams(16000, 1, AudioFormat.WAV)
    payload = AudioPayload.from_path("/tmp/example.wav", params)
    assert payload is not None


def test_voice_ref_named_id_cloned_from() -> None:
    assert VoiceRef.named("alloy") is not None
    assert VoiceRef.id("21-char-id-foo") is not None

    payload = AudioPayload.from_bytes(b"abc", AudioParams(16000, 1, AudioFormat.WAV))
    assert VoiceRef.cloned_from(payload) is not None


def test_speech_batch_tts_construction() -> None:
    """TTS modality — SpeechBatch constructor + estimated_characters."""
    batch = SpeechBatch(
        request_id="r1",
        model="tts-1",
        text="hello world",
        voice=VoiceRef.named("alloy"),
        options=SynthOptions(),
        stream=True,
        emit_alignment=False,
    )
    assert batch.request_id == "r1"
    assert batch.model == "tts-1"
    assert batch.text == "hello world"
    assert batch.stream is True
    assert batch.emit_alignment is False
    assert batch.estimated_characters == len("hello world")


def test_audio_batch_transcribe_construction() -> None:
    """STT modality — AudioBatch.transcribe constructor."""
    params = AudioParams(16000, 1, AudioFormat.WAV)
    payload = AudioPayload.from_bytes(b"\x00" * 32, params)
    batch = AudioBatch.transcribe(
        request_id="r2",
        model="whisper-1",
        payload=payload,
        options=TranscribeOptions(language="en", word_timestamps=True),
        stream=False,
        estimated_units=4,
    )
    assert batch.request_id == "r2"
    assert batch.model == "whisper-1"
    assert batch.stream is False
    assert batch.estimated_units == 4


def test_audio_batch_audio2face_construction() -> None:
    """A2F modality — AudioBatch.audio2face constructor."""
    params = AudioParams(16000, 1, AudioFormat.PCM16_LE)
    payload = AudioPayload.from_bytes(b"\x00" * 32, params)
    batch = AudioBatch.audio2face(
        request_id="r3",
        model="audio2face-3d",
        payload=payload,
        options=A2FOptions(fps=30),
        stream=True,
        estimated_units=10,
    )
    assert batch.request_id == "r3"
    assert batch.model == "audio2face-3d"
    assert batch.stream is True
    assert batch.estimated_units == 10


def test_synth_options_optional_arguments() -> None:
    """SynthOptions accepts all-None construction (no required args)."""
    opts = SynthOptions()
    assert opts is not None
    opts2 = SynthOptions(format=AudioFormat.WAV, sample_rate_hz=22050, speed=1.0)
    assert opts2 is not None


def test_transcribe_options_optional_arguments() -> None:
    opts = TranscribeOptions()
    assert opts is not None
    opts2 = TranscribeOptions(language="en", interim_results=True, diarize=True)
    assert opts2 is not None


def test_a2f_options_optional_arguments() -> None:
    opts = A2FOptions()
    assert opts is not None
    opts2 = A2FOptions(fps=60, emotion="happy")
    assert opts2 is not None


def test_audio_submodule_facade_re_exports() -> None:
    """The pure-Python facade re-exports everything from the native submodule."""
    import atomr_infer.audio as facade
    expected = {
        "AudioFormat", "AudioParams", "AudioPayload", "VoiceRef",
        "TranscribeOptions", "SynthOptions", "A2FOptions",
        "SpeechBatch", "AudioBatch",
        "WordTiming", "Viseme", "AlignmentDelta",
        "TranscriptChunk", "SpeechChunk", "BlendshapeChunk",
    }
    missing = expected - set(facade.__all__)
    assert not missing, f"missing from facade: {sorted(missing)}"
