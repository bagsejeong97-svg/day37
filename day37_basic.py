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
    # JSONL 기록 (표준 open 함수 사용으로 metadata_errors 충돌 방지)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
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
                cur.