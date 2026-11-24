#!/usr/bin/env python3
"""
محرك استنساخ الصوت - Voice Cloning Engine
-------------------------------------------
- مجاني بالكامل، يعمل محلياً، بدون أي مفتاح API
- النموذج: Coqui XTTS-v2 (يدعم العربية والإنجليزية و15 لغة أخرى)
- يحتاج مقطعاً مرجعياً من 6-30 ثانية (كلما تعددت المقاطع النظيفة زادت الدقة)

التثبيت:
    pip install -r requirements.txt

التشغيل (واجهة ويب):
    python app.py

التشغيل (سطر أوامر):
    python app.py --text "مرحباً بكم" --ref my_voice.wav --lang ar --out result.wav

تنبيه أخلاقي وقانوني: استخدم أصوات أشخاص وافقوا على ذلك فقط.
رخصة النموذج (CPML) للاستخدام غير التجاري.
"""
import argparse
import os
import sys
import tempfile
import uuid

# قبول رخصة Coqui تلقائياً (غير تجارية) لتجنب السؤال التفاعلي
os.environ.setdefault("COQUI_TOS_AGREED", "1")

import torch
import torchaudio

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
SAMPLE_RATE = 24000

LANGUAGES = {
    "العربية": "ar", "English": "en", "Español": "es", "Français": "fr",
    "Deutsch": "de", "Italiano": "it", "Português": "pt", "Polski": "pl",
    "Türkçe": "tr", "Русский": "ru", "Nederlands": "nl", "Čeština": "cs",
    "中文": "zh-cn", "日本語": "ja", "한국어": "ko", "Magyar": "hu", "हिन्दी": "hi",
}

_model = None


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model():
    """تحميل النموذج مرة واحدة (أول تشغيل يحمّل ~1.8GB تلقائياً)."""
    global _model
    if _model is None:
        from TTS.api import TTS
        print(f"[+] تحميل النموذج على الجهاز: {get_device()} ...")
        _model = TTS(MODEL_NAME).to(get_device())
    return _model


def prepare_reference(path: str, out_dir: str) -> str:
    """تنظيف المقطع المرجعي: أحادي القناة، 24kHz، قص الصمت، تطبيع الصوت."""
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True)  # mono
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    # قص الصمت من البداية والنهاية (حسب الطاقة)
    energy = wav.abs().squeeze(0)
    thr = energy.max() * 0.02
    idx = (energy > thr).nonzero()
    if len(idx) > 0:
        wav = wav[:, idx[0].item(): idx[-1].item() + 1]

    # حد أقصى 30 ثانية
    wav = wav[:, : SAMPLE_RATE * 30]

    # تطبيع القمة
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak * 0.95

    duration = wav.shape[1] / SAMPLE_RATE
    if duration < 3:
        raise ValueError(f"المقطع المرجعي قصير جداً ({duration:.1f}s). استخدم 6-30 ثانية.")

    out = os.path.join(out_dir, f"ref_{uuid.uuid4().hex[:8]}.wav")
    torchaudio.save(out, wav, SAMPLE_RATE)
    return out


def clone(text: str, ref_files, lang: str = "ar", speed: float = 1.0,
          temperature: float = 0.65, repetition_penalty: float = 5.0,
          out_path: str | None = None) -> str:
    """توليد كلام بصوت المتحدث المرجعي. يرجع مسار ملف WAV."""
    if not text or not text.strip():
        raise ValueError("النص فارغ.")
    if isinstance(ref_files, str):
        ref_files = [ref_files]
    if not ref_files:
        raise ValueError("يجب رفع مقطع صوتي مرجعي واحد على الأقل.")

    tmp = tempfile.mkdtemp(prefix="vc_")
    refs = [prepare_reference(p, tmp) for p in ref_files]
    out_path = out_path or os.path.join(tmp, "output.wav")

    get_model().tts_to_file(
        text=text.strip(),
        speaker_wav=refs,
        language=lang,
        file_path=out_path,
        split_sentences=True,
        speed=speed,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )
    return out_path


# ---------------------------- واجهة الويب ----------------------------
def launch_ui():
    import gradio as gr

    def run(text, files, lang_name, speed, temperature, consent):
        if not consent:
            raise gr.Error("يجب الإقرار بأن لديك إذن صاحب الصوت.")
        if not files:
            raise gr.Error("ارفع مقطعاً صوتياً مرجعياً.")
        paths = [f if isinstance(f, str) else f.name for f in files]
        try:
            return clone(text, paths, LANGUAGES[lang_name], speed, temperature)
        except Exception as e:
            raise gr.Error(str(e))

    with gr.Blocks(title="محرك استنساخ الصوت", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎙️ محرك استنساخ الصوت\nمحلي · مجاني · بدون API")
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="النص المراد نطقه", lines=6, rtl=True,
                                  placeholder="اكتب النص هنا...")
                files = gr.File(label="مقاطع صوتية مرجعية (6-30 ثانية، نظيفة بلا موسيقى)",
                                file_count="multiple", file_types=["audio"])
                lang = gr.Dropdown(list(LANGUAGES), value="العربية", label="اللغة")
                with gr.Accordion("إعدادات متقدمة", open=False):
                    speed = gr.Slider(0.7, 1.4, value=1.0, step=0.05, label="السرعة")
                    temp = gr.Slider(0.1, 1.0, value=0.65, step=0.05,
                                     label="التنوع (أقل = أثبت، أعلى = أكثر تعبيراً)")
                consent = gr.Checkbox(label="أقرّ بأنني أملك إذن صاحب الصوت لاستنساخه")
                btn = gr.Button("توليد الصوت", variant="primary")
            with gr.Column():
                out = gr.Audio(label="النتيجة", type="filepath")
        btn.click(run, [text, files, lang, speed, temp, consent], out)

    # 0.0.0.0 ضروري ليعمل داخل GitHub Codespaces / الحاويات
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)


def main():
    p = argparse.ArgumentParser(description="Voice cloning engine (XTTS-v2)")
    p.add_argument("--text", help="النص")
    p.add_argument("--ref", nargs="+", help="ملف/ملفات صوتية مرجعية")
    p.add_argument("--lang", default="ar", help="رمز اللغة (ar, en, fr, ...)")
    p.add_argument("--out", default="output.wav")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.65)
    p.add_argument("--yes-i-have-consent", action="store_true",
                   help="أقرّ بأن لدي إذن صاحب الصوت")
    a = p.parse_args()

    if not a.text:
        launch_ui()
        return
    if not a.yes_i_have_consent:
        sys.exit("أضف --yes-i-have-consent للإقرار بأن لديك إذن صاحب الصوت.")
    if not a.ref:
        sys.exit("حدد --ref مقطعاً مرجعياً.")
    path = clone(a.text, a.ref, a.lang, a.speed, a.temperature, a.out)
    print(f"[✓] تم الحفظ: {path}")


if __name__ == "__main__":
    main()
