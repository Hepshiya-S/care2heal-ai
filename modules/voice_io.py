import os
import shutil
import time
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import imageio_ffmpeg
from gtts import gTTS
from agents.rag_pipeline import llm

# imageio_ffmpeg ships a versioned filename, but Whisper's internal code
# calls the plain command "ffmpeg" — create a renamed copy so that lookup
# succeeds, then add the folder to PATH for this process only.
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(ffmpeg_exe)
ffmpeg_alias = os.path.join(ffmpeg_dir, "ffmpeg.exe")

if not os.path.exists(ffmpeg_alias):
    shutil.copy(ffmpeg_exe, ffmpeg_alias)

os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

whisper_model = whisper.load_model("base")


def record_audio(duration: int = 8, filename: str = "temp_input.wav", sample_rate: int = 16000) -> str:
    print(f"Recording for {duration} seconds... speak now.")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    write(filename, sample_rate, audio)
    print("Recording finished.")
    return filename


def transcribe_and_translate_to_english(audio_path: str) -> str:
    result = whisper_model.transcribe(audio_path, task="translate")
    return result["text"].strip()


def translate_to_english(text: str) -> str:
    prompt = (
        "Translate the following text into English. If it is already in English, "
        "return it unchanged. Return ONLY the translated text, nothing else.\n\n"
        f"Text: {text}"
    )
    response = llm.invoke(prompt)
    return response.content.strip()


def translate_and_speak(english_text: str, target_lang_code: str, target_lang_name: str, output_path: str = None) -> str:
    if output_path is None:
        output_path = f"response_{int(time.time() * 1000)}.mp3"

    if target_lang_code != "en":
        prompt = (
            f"Translate the following text into {target_lang_name}. "
            "Return ONLY the translated text, nothing else.\n\n"
            f"Text: {english_text}"
        )
        response = llm.invoke(prompt)
        text_to_speak = response.content.strip()
    else:
        text_to_speak = english_text

    tts = gTTS(text=text_to_speak, lang=target_lang_code)
    tts.save(output_path)
    return text_to_speak


if __name__ == "__main__":
    audio_file = record_audio(duration=8)
    english_question = transcribe_and_translate_to_english(audio_file)
    print(f"\nYou said (translated to English): {english_question}\n")
    translate_and_speak(english_question, target_lang_code="ta", target_lang_name="Tamil", output_path="test_response.mp3")