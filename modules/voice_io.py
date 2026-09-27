import os
import shutil
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import imageio_ffmpeg
from gtts import gTTS
from agents.rag_pipeline import llm

# imageio_ffmpeg ships a versioned filename (e.g. ffmpeg-win-x86_64-v7.1.exe),
# but Whisper's internal code specifically calls the plain command "ffmpeg".
# We create a renamed copy so that exact lookup succeeds, then add the
# folder to PATH for this process only — no system-level install needed.
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(ffmpeg_exe)
ffmpeg_alias = os.path.join(ffmpeg_dir, "ffmpeg.exe")

if not os.path.exists(ffmpeg_alias):
    shutil.copy(ffmpeg_exe, ffmpeg_alias)

os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

whisper_model = whisper.load_model("base")


def record_audio(duration: int = 5, filename: str = "temp_input.wav", sample_rate: int = 16000) -> str:
    """
    Records from the default microphone for `duration` seconds, saves as WAV.
    16kHz matches what Whisper expects internally.
    """
    print(f"Recording for {duration} seconds... speak now.")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    write(filename, sample_rate, audio)
    print("Recording finished.")
    return filename


def transcribe_and_translate_to_english(audio_path: str) -> str:
    """
    Whisper's 'translate' task transcribes speech in ANY supported language
    and returns English text directly — one step, no separate translation call
    needed for the input side, even for Tamil, Hindi, etc.
    """
    result = whisper_model.transcribe(audio_path, task="translate")
    return result["text"].strip()


def translate_and_speak(english_text: str, target_lang_code: str, target_lang_name: str, output_path: str = "response.mp3") -> str:
    """
    Translates an English answer into the target language (via our existing LLM),
    then converts it to speech using gTTS.
    """
    if target_lang_code != "en":
        prompt = f"""Translate the following text into {target_lang_name}.
Return ONLY the translated text, nothing else.

Text: {english_text}"""
        response = llm.invoke(prompt)
        text_to_speak = response.content.strip()
    else:
        text_to_speak = english_text

    tts = gTTS(text=text_to_speak, lang=target_lang_code)
    tts.save(output_path)
    print(f"Spoken response saved to {output_path}")
    return text_to_speak


if __name__ == "__main__":
    audio_file = record_audio(duration=5)
    english_question = transcribe_and_translate_to_english(audio_file)
    print(f"\nYou said (translated to English): {english_question}\n")

    # Quick demo: speak the transcribed text back in Tamil
    translate_and_speak(english_question, target_lang_code="ta", target_lang_name="Tamil")