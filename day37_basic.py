# day37_basic.py
# 주제: 캠퍼스 시설 안내 음성 도우미
import asyncio # 비동기 TTS 함수를 실행
import json # 실행 기록을 JSONL 형식으로 저장
import os # DATABASE_URL 환경변수 조회
import tempfile # STT 입력을 위한 임시 WAV 파일 생성
import time # 처리 단계별 실행 시간 측정
import uuid # 요청별 고유 ID 생성
import wave # WAV 파일 헤더와 PCM 형식 검사
from datetime import datetime, timezone # UTC 시각 생성
from pathlib import Path # 파일 경로 처리

import streamlit as st # 웹 기반 사용자 인터페이스
from dotenv import load_dotenv

load_dotenv() # .env 로딩

FACILITIES = {
    "도서관": "도서관은 본관 옆 건물이며 평일 9시부터 18시까지 운영합니다.",
    "학생식당": "학생식당은 학생회관 1층에 있으며 11시부터 14시까지 운영합니다.",
    "취업지원센터": "취업지원센터는 학생회관 2층이며 취업 상담을 제공합니다.",
    "행정실": "행정실은 본관 1층이며 각종 증명서 발급을 안내합니다.",
}
MAX_BYTES = 10 * 1024 * 1024

# 실행이력을 저장할 jsonl파일 경로
LOG_PATH = Path(__file__).with_name('day37_runs.jsonl')

# 사용할 Whisper모델 크기
MODEL_SIZE = 'base'

# Edge TTS에서 사용하는 한국어 여성 음성 모델 ID
VOICE_ID = 'ko-KR-SunHiNeural'
SQL_DIR = Path(__file__).resolve().parent / 'sql'


if not SQL_DIR.exists(): # 디렉토리 없다면
    SQL_DIR.mkdir(parents=True, exist_ok=True) # 새로 생성

def load_sql(filename):
    path = (SQL_DIR / filename).resolve() # SQL파일의 서버상의 실제경로
    if path.parent != SQL_DIR.resolve() or path.suffix != '.sql':
        raise ValueError('sql폴더의 .sql파일명만 지정하세요.')
    return path.read_text(encoding='utf-8') # SQL파일내의 문자열을 리턴

def validate_wav(audio):
    import io
    try:
        with wave.open(io.BytesIO(audio), 'rb') as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            frames = wav.getnframes()
            compression = wav.getcomptype()
    except (wave.Error, EOFError, OSError) as e:
        raise ValueError('유효한 비압축 PCM WAV 파일을 입력해 주세요.') from e

    if compression != 'NONE' or channels not in (1, 2) or sample_rate < 0 or frames <= 0:
        raise ValueError('모노/스테레오 비압축 PCM WAV 파일이 필요합니다.')

    return {
        'channels': channels,
        'sample_rate': sample_rate,
        'bit_depth': sample_width * 8,
        'duration_sec': round(frames / sample_rate, 2)
    }

@st.cache_resource(show_spinner='Wisper 모델을 로딩하고 있습니다...')
def load_stt_model(model_size):
    from faster_whisper import WhisperModel
    return WhisperModel(model_size, device='CPU', compute_type='int8')

def transcribe_wav(audio, model_size):
    model = load_stt_model(model_size)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        f.write(audio)
        wav_path = Path(f.name)

    try:
        segments, info = model.transcribe(
            str(wav_path),
            language='ko',
            vad_filter=True
        )
        transcript = ''.join(segment.text.strip() for segment in segments).strip()
        return transcript
    finally:
        wav_path.unlink(missing_ok=True)

# 답변 생성 함수
def build_reply(transcript):
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError('질문 내용을 다시 확인 해주세요.')

    clean = ''.join(transcript.split())

    for key, value in FACILITIES.items():
        if key in clean:
            return value

    facility_list = ", ".join(FACILITIES.keys())
    return f"죄송합니다. 질문하신 내용을 이해하지 못했어요. 다음 시설 중 하나에 대해 질문해 주세요: {facility_list}"

async def generate_tts_mp3(text, voice):
    import edge_tts
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
        mp3_path = Path(f.name)
    try:
        await edge_tts.Communicate(text=text, voice=voice).save(str(mp3_path))
        return mp3_path.read_bytes()
    finally:
        mp3_path.unlink(missing_ok=True)

def synthesize_speech(text, voice_id):
    mp3_bytes = asyncio.run(generate_tts_mp3(text, voice_id))
    return mp3_bytes, 'audio/mpeg'

def store_meta(meta):
    # JSONL 기록
    with LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(json.dumps(meta, ensure_ascii=False) + '\n')

    database_url = os.getenv('DATABASE_URL', '').strip()
    if not database_url:
        return

    try:
        import psycopg
        with psycopg.connect(database_url, connect_timeout=5) as conn:
            sql = load_sql('create_table.sql')
            sql1 = load_sql('insert.sql')
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute(sql1, (
                    meta['request_id'], meta['created_at'], meta['model_version'],
                    meta['status'], meta['stt_ms'], meta['reply_ms'],
                    meta['tts_ms'], meta['total_ms']
                ))
            # 💡 핵심 수정: 데이터베이스 변경 사항을 영구 반영하기 위해 반드시 commit 필요!
            conn.commit()
    except Exception as e:
        # DB 저장 실패 시 원인을 명확하게 화면에 노출하도록 수정
        st.warning(f'DB 실행 이력 저장 실패: {type(e).__name__}: {e}')

def run_pipeline(audio, model_size, voice_id):
    request_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    model_version = f'whisper-{model_size}/edge-tts/{voice_id}'

    marks = {'stt_ms': 0.0, 'reply_ms': 0.0, 'tts_ms': 0.0}
    started = time.perf_counter()

    result = None
    error = None
    try:
        t0 = time.perf_counter()
        transcript = transcribe_wav(audio, model_size)
        marks['stt_ms'] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        reply = build_reply(transcript)
        marks['reply_ms'] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        output, mime = synthesize_speech(reply, voice_id)
        marks['tts_ms'] = (time.perf_counter() - t0) * 1000

        result = (transcript, reply, output, mime)

    except Exception as e:
        error = e

    elapsed = (time.perf_counter() - started) * 1000

    meta = {
        'request_id': request_id,
        'created_at': created_at,
        'model_version': model_version,
        'status': 'error' if error else 'success',
        'stt_ms': round(marks.get('stt_ms', 0.0), 2),
        'reply_ms': round(marks.get('reply_ms', 0.0), 2),
        'tts_ms': round(marks.get('tts_ms', 0.0), 2),
        'total_ms': round(elapsed, 2),
    }

    store_meta(meta)

    return result, marks, meta, error

def main():
    st.set_page_config(
        page_title='캠퍼스 시설 안내 음성 도우미',
        layout='centered'
    )

    st.title('캠퍼스 시설별 안내 음성 도우미')
    st.caption('WAV파일 > faster-whisper STT > 규칙 기반 응답 > edge-tts 음성 합성')

    uploaded = st.audio_input('마이크로 한국어 음성을 녹음하세요')

    model_size = st.selectbox(
        'STT 모델',
        ['tiny', 'base', 'small', 'medium'],
        index=1
    )

    voice_id = st.selectbox(
        'TTS 목소리',
        ['ko-KR-SunHiNeural', 'ko-KR-InJoonNeural']
    )

    submitted = st.button('음성 파이프라인 실행')

    if not submitted:
        return

    if uploaded is None:
        st.error('먼저 음성 녹음을 완료해주세요')
        return

    audio = uploaded.getvalue()

    if not audio or len(audio) > MAX_BYTES:
        st.error(f'WAV 파일은 0바이트보다 크고 {MAX_BYTES}바이트 이하여야 합니다.')
        return

    try:
        audio_info = validate_wav(audio)
    except ValueError as e:
        st.error(str(e))
        return

    st.subheader('입력 WAV 파일')
    st.audio(audio, format='audio/wav')
    st.json(audio_info)

    with st.spinner('실제 음성 인식 및 음성 합성 처리 중...'):
        result, marks, meta, error = run_pipeline(audio, model_size, voice_id)

    if error:
        st.error(f'처리 오류: {type(error).__name__}: {error}')
    else:
        transcript, reply, output, mime = result

        st.subheader('STT 인식 결과')
        st.write(transcript or '(인식된 음성이 없습니다)')

        st.subheader('응답 문장')
        st.write(reply)

        st.subheader('TTS 합성 음성')
        st.audio(output, format=mime)

        st.download_button('TTS MP3 저장', output, 'reply.mp3', mime=mime)

    st.metric('총 처리시간', f"{meta['total_ms']:.0f} ms")

    st.bar_chart(marks)
    st.json(meta)

if __name__ == '__main__':
    main()