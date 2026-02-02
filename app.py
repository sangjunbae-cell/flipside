import streamlit as st
import requests
import re
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

# --- 페이지 설정 ---
st.set_page_config(page_title="Veritas Lens", page_icon="👁️", layout="wide")

# --- CSS 커스텀 ---
st.markdown("""
    <style>
    .main-title {font-size: 3rem; font-weight: 800; color: #111827; letter-spacing: -0.05rem;}
    .sub-title {font-size: 1.2rem; color: #6B7280; margin-bottom: 2rem;}
    div.stButton > button {
        background-color: #2563EB; color: white; border-radius: 8px; 
        padding: 0.5rem 1rem; font-weight: bold; border: none;
        width: 100%;
    }
    div.stButton > button:hover {background-color: #1D4ED8;}
    </style>
""", unsafe_allow_html=True)

# --- 사이드바: API 설정 ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # OpenAI & Tavily
    if "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    else:
        openai_api_key = st.text_input("OpenAI API Key", type="password")

    if "TAVILY_API_KEY" in st.secrets:
        tavily_api_key = st.secrets["TAVILY_API_KEY"]
    else:
        tavily_api_key = st.text_input("Tavily API Key", type="password")
        
    # RapidAPI Key
    st.markdown("---")
    st.subheader("📺 YouTube Unlocker")
    if "RAPIDAPI_KEY" in st.secrets:
        rapid_api_key = st.secrets["RAPIDAPI_KEY"]
    else:
        rapid_api_key = st.text_input("RapidAPI Key", type="password")
        
    st.info("👁️ **Veritas Lens**는 최신 AI와 검색 기술을 결합하여 콘텐츠의 진실을 탐구합니다.")

# --- 공통 함수 ---

def get_llm(openai_key):
    # 가성비 모델 gpt-4o-mini 사용 (필요시 gpt-4o로 변경 가능)
    return ChatOpenAI(temperature=0, openai_api_key=openai_key, model_name="gpt-4o-mini")

def get_search_tool(tavily_key):
    return TavilySearchResults(tavily_api_key=tavily_key, k=3)

# 🛠️ [Helper] 유튜브 메타데이터 가져오기
def get_youtube_metadata(url):
    try:
        oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
        response = requests.get(oembed_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "title": data.get("title", "YouTube Video"),
                "author": data.get("author_name", "Unknown Channel"),
                "thumbnail": data.get("thumbnail_url", "https://img.youtube.com/vi/default/hqdefault.jpg"),
                "url": url
            }
    except:
        pass
    return {"title": "분석된 유튜브 영상", "author": "YouTube", "thumbnail": "https://img.youtube.com/vi/default/hqdefault.jpg", "url": url}

# 🚀 [Core] RapidAPI 자막 추출
def get_transcript_via_api(video_url, api_key):
    url = "https://youtube-transcript3.p.rapidapi.com/api/transcript-with-url"
    querystring = {"url": video_url, "flat_text": "true", "lang": "ko"}
    headers = {"x-rapidapi-key": api_key, "x-rapidapi-host": "youtube-transcript3.p.rapidapi.com"}

    try:
        response = requests.get(url, headers=headers, params=querystring, timeout=20)
        
        # 한국어 실패 시 영어 재시도
        if response.status_code != 200:
            querystring["lang"] = "en"
            response = requests.get(url, headers=headers, params=querystring, timeout=20)
            
        if response.status_code != 200:
            raise Exception(f"API 호출 실패 ({response.status_code})")

        data = response.json()
        
        if "transcript" in data:
            if len(data["transcript"]) < 50:
                raise Exception("자막 내용이 너무 짧거나 없습니다.")
            return data["transcript"][:15000]
        elif "message" in data:
            raise Exception(f"API 에러: {data['message']}")
        else:
            return str(data)[:15000]

    except Exception as e:
        raise Exception(f"자막 추출 중 오류 발생: {e}")

# ---------------------------------------------------------
# 🔥 [NEW] RAG 심층 분석 파이프라인
# ---------------------------------------------------------
def deep_analyze_with_search(text, llm, search_tool):
    """
    1. 텍스트에서 검증 필요한 주장 추출
    2. Tavily로 웹 검색 수행
    3. 주장과 검색 결과를 종합하여 팩트체크 리포트 생성
    """
    
    # 1단계: 검증할 핵심 키워드/주장 추출
    with st.spinner("🕵️‍♀️ 1단계: 검증이 필요한 핵심 주장을 선별 중..."):
        extraction_prompt = PromptTemplate.from_template("""
        다음 텍스트에서 사실 검증이 필요한 핵심 주장이나 키워드 3가지를 추출해줘.
        검색 엔진에 입력할 쿼리 형태로 만들어줘.
        
        [텍스트]
        {text}
        
        [출력 형식]
        - 검색쿼리1
        - 검색쿼리2
        - 검색쿼리3
        """)
        claims_result = llm.invoke(extraction_prompt.format(text=text[:10000])).content
        queries = [line.replace("-", "").strip() for line in claims_result.split('\n') if line.strip().startswith("-")]

    # 2단계: 웹 검색 수행 (Grounding)
    search_context = ""
    with st.spinner(f"🌐 2단계: 웹에서 팩트 확인 중... ({len(queries)}건)"):
        for query in queries[:3]: # 비용 절약을 위해 최대 3개만
            try:
                search_results = search_tool.invoke(query)
                # 검색 결과 요약
                evidence = "\n".join([f"- 출처({res['url']}): {res['content'][:200]}" for res in search_results])
                search_context += f"\n[검색 키워드: {query}]\n{evidence}\n"
            except Exception as e:
                pass

    # 3단계: 최종 종합 분석 (RAG)
    with st.spinner("🧠 3단계: 증거를 바탕으로 심층 리포트 작성 중..."):
        final_prompt = PromptTemplate.from_template("""
        당신은 냉철한 미디어 비평가입니다. 
        제공된 [원본 텍스트]와 [외부 검색 증거]를 비교 분석하여 보고서를 작성하세요.
        
        [원본 텍스트]
        {text}
        
        [외부 검색 증거 (Fact Check Materials)]
        {context}
        
        [필수 요청사항]
        1. 핵심요약: 3가지 (각 1문장)
        2. 신뢰도점수: 0~100점 (검색 증거와 일치하면 높게, 다르면 낮게)
        3. 화자성향: 1문장 요약 (원본이 팩트를 어떻게 왜곡하거나 강조하는지 분석)
        4. AI코멘트: 이 콘텐츠를 받아들이는 시청자를 위한 조언
        5. 팩트체크: 반드시 [외부 검색 증거]를 기반으로 판단할 것.
        
        [출력 형식]
        SUMMARY:
        - 요약1
        - 요약2
        - 요약3
        SCORE: 75
        STANCE: 성향
        COMMENT: 코멘트
        CLAIMS:
        - 주장1 | 사실/거짓/의견 | 이유
        - 주장2 | 사실/거짓/의견 | 이유
        - 주장3 | 사실/거짓/의견 | 이유
        """)
        
        return llm.invoke(final_prompt.format(text=text[:10000], context=search_context)).content

# ---------------------------------------------------------
# 🎨 [UX 1] 유튜브 분석 함수 (Updated)
# ---------------------------------------------------------
def analyze_youtube(url, llm, search, api_key):
    meta = get_youtube_metadata(url)
    full_text = ""
    
    with st.spinner("🎧 영상 데이터를 가져오는 중... (RapidAPI)"):
        try:
            full_text = get_transcript_via_api(url, api_key)
        except Exception as e:
            st.error(f"❌ 분석 중단: {e}")
            return

    # --- RAG 심층 분석 적용 ---
    try:
        result = deep_analyze_with_search(full_text, llm, search)
    except Exception as e:
        st.error(f"AI 분석 오류: {e}")
        return
    # -----------------------

    # 파싱
    summary_list = []
    score = 50
    stance = "분석 불가"
    comment = "정보 없음"
    claims_data = []

    current_section = None
    for line in result.split('\n'):
        line = line.strip()
        if not line: continue
        if "SUMMARY:" in line: current_section = "SUMMARY"; continue
        if "SCORE:" in line: 
            try: score = int(re.findall(r'\d+', line)[0])
            except: score = 50; continue
        if "STANCE:" in line: stance = line.replace("STANCE:", "").strip(); continue
        if "COMMENT:" in line: comment = line.replace("COMMENT:", "").strip(); continue
        if "CLAIMS:" in line: current_section = "CLAIMS"; continue
        
        if current_section == "SUMMARY" and line.startswith("-"): summary_list.append(line.replace("-", "").strip())
        if current_section == "CLAIMS" and line.startswith("-"):
            parts = line.replace("-", "").strip().split("|")
            if len(parts) >= 3: claims_data.append({"claim": parts[0].strip(), "type": parts[1].strip(), "reason": parts[2].strip()})

    # HTML 조립
    summary_html = "".join([f'<li class="flex items-start"><div class="bg-blue-600 text-white rounded-full w-5 h-5 flex items-center justify-center text-xs mr-3 mt-0.5 flex-shrink-0">{i}</div><span>{t}</span></li>' for i, t in enumerate(summary_list, 1)])
    
    claims_html = ""
    for item in claims_data:
        if "사실" in item['type']: theme = ("text-green-500", "bg-green-50", "border-green-500", "text-green-800", "사실 (Fact)", "fa-check-circle")
        elif "거짓" in item['type']: theme = ("text-red-500", "bg-red-50", "border-red-500", "text-red-800", "거짓/오류 (False)", "fa-times-circle")
        else: theme = ("text-yellow-500", "bg-yellow-50", "border-yellow-500", "text-yellow-800", "의견/전망 (Opinion)", "fa-scale-balanced")
        
        claims_html += f"""
        <div class="flex space-x-4"><div class="mt-1"><i class="fa-solid {theme[5]} {theme[0]} text-xl"></i></div>
        <div><h4 class="font-bold text-gray-900 mb-1">"{item['claim']}"</h4>
        <p class="text-sm text-gray-700 {theme[1]} p-3 rounded-lg border-l-4 {theme[2]}"><strong class="{theme[3]}">{theme[4]}</strong><br>{item['reason']}</p></div></div>"""

    score_theme = ("text-green-600", "border-green-400", "bg-green-50", "신뢰도 높음") if score >= 70 else \
                  ("text-yellow-700", "border-yellow-400", "bg-yellow-50", "주의 필요") if score >= 40 else \
                  ("text-red-600", "border-red-400", "bg-red-50", "신뢰도 낮음")

    final_html = f"""
    <!DOCTYPE html><html lang="ko"><head><script src="https://cdn.tailwindcss.com"></script><link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet"><style>@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');body {{ font-family: 'Noto Sans KR', sans-serif; background-color: transparent; }}.card {{ background: #ffffff; border-radius: 16px; padding: 24px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); border: 1px solid #f3f4f6; margin-bottom: 20px; }}</style></head><body>
    <div class="max-w-4xl mx-auto space-y-6">
        <div class="card flex items-start space-x-4"><img src="{meta['thumbnail']}" class="w-32 h-auto rounded-xl shadow-sm"><div class="flex-1"><h2 class="text-xl font-bold text-gray-900 leading-tight mb-2">{meta['title']}</h2><p class="text-sm text-gray-500 mb-2"><i class="fa-brands fa-youtube mr-1 text-red-600"></i> {meta['author']}</p><a href="{meta['url']}" target="_blank" class="text-sm text-blue-600 hover:underline font-medium">영상 보러가기 <i class="fa-solid fa-external-link-alt text-xs ml-1"></i></a></div></div>
        <div class="card"><h3 class="text-lg font-bold text-gray-900 mb-4 flex items-center"><i class="fa-solid fa-list-check mr-2 text-blue-600"></i> 핵심 3줄 요약</h3><ul class="space-y-3 text-gray-700">{summary_html}</ul></div>
        <div class="card"><h3 class="text-lg font-bold text-gray-900 mb-6 flex items-center"><i class="fa-solid fa-chart-pie mr-2 text-blue-600"></i> 심층 분석</h3><div class="grid grid-cols-1 md:grid-cols-2 gap-8"><div class="flex flex-col items-center justify-center p-4 bg-gray-50 rounded-xl"><p class="text-sm font-medium text-gray-500 mb-3">AI 신뢰도 점수</p><div class="relative w-24 h-24 flex items-center justify-center rounded-full border-8 {score_theme[1]} {score_theme[2]} mb-2"><span class="text-3xl font-bold {score_theme[0]}">{score}</span></div><p class="font-bold {score_theme[0]}">{score_theme[3]}</p></div><div class="flex flex-col justify-center"><div class="mb-4"><p class="text-sm font-medium text-gray-500 mb-1">🗣️ 화자 성향 분석</p><p class="text-gray-800 font-semibold text-lg">{stance}</p></div><div class="bg-blue-50 p-4 rounded-lg border border-blue-100"><p class="text-xs font-bold text-blue-800 mb-1"><i class="fa-solid fa-robot"></i> AI Insight</p><p class="text-sm text-blue-900 leading-relaxed">{comment}</p></div></div></div></div>
        <div class="card"><h3 class="text-lg font-bold text-gray-900 mb-6 flex items-center"><i class="fa-solid fa-magnifying-glass mr-2 text-blue-600"></i> 주요 주장 팩트체크</h3><div class="space-y-6">{claims_html}</div></div>
    </div></body></html>"""
    st.markdown(final_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# 🎨 [UX 2] 뉴스 분석 함수 (기존 유지)
# ---------------------------------------------------------
def analyze_article(url, llm, search):
    try:
        with st.spinner("📰 기사 본문을 읽어오는 중..."):
            loader = WebBaseLoader(url)
            docs = loader.load()
            content = docs[0].page_content[:12000]
            title = docs[0].metadata.get('title', '뉴스 기사 분석')
            domain = url.split("//")[-1].split("/")[0].replace("www.", "")
    except Exception as e:
        st.error(f"기사를 읽어올 수 없습니다: {e}")
        return

    # 뉴스 분석도 RAG를 쓰고 싶다면 여기도 deep_analyze_with_search(content, llm, search)로 교체 가능
    # 현재는 기존 로직 유지
    with st.spinner("⚖️ Veritas Lens가 기사의 이면을 파헤치고 있습니다..."):
        analysis_prompt = PromptTemplate.from_template("""
        다음 뉴스 기사를 분석해서 구조화된 데이터를 만들어줘.
        [기사 본문] {text}
        [요청사항]
        1. 핵심요약: 3가지 요약 (각 1문장)
        2. 자극성점수: 0~100점 (높을수록 자극적)
        3. 프레이밍: 1문장 요약
        4. AI코멘트: 1문장
        5. 팩트체크: 3가지 판별
        [출력 형식]
        SUMMARY:
        - 요약1
        - 요약2
        - 요약3
        SCORE: 85
        FRAMING: 프레이밍
        COMMENT: 코멘트
        CLAIMS:
        - 주장1 | 사실/거짓/의견 | 이유
        - 주장2 | 사실/거짓/의견 | 이유
        - 주장3 | 사실/거짓/의견 | 이유
        """)
        
        try:
            result = llm.invoke(analysis_prompt.format(text=content)).content
        except Exception as e:
            st.error(f"AI 분석 오류: {e}")
            return

        # 파싱
        summary_list = []
        score = 50
        framing = "분석 불가"
        comment = "정보 없음"
        claims_data = []
        current_section = None
        
        for line in result.split('\n'):
            line = line.strip()
            if not line: continue
            if "SUMMARY:" in line: current_section = "SUMMARY"; continue
            if "SCORE:" in line: 
                try: score = int(re.findall(r'\d+', line)[0])
                except: score = 50; continue
            if "FRAMING:" in line: framing = line.replace("FRAMING:", "").strip(); continue
            if "COMMENT:" in line: comment = line.replace("COMMENT:", "").strip(); continue
            if "CLAIMS:" in line: current_section = "CLAIMS"; continue
            
            if current_section == "SUMMARY" and line.startswith("-"): summary_list.append(line.replace("-", "").strip())
            if current_section == "CLAIMS" and line.startswith("-"):
                parts = line.replace("-", "").strip().split("|")
                if len(parts) >= 3: claims_data.append({"claim": parts[0].strip(), "type": parts[1].strip(), "reason": parts[2].strip()})

        # HTML 조립
        summary_html = "".join([f'<li class="flex items-start"><div class="bg-indigo-600 text-white rounded-full w-5 h-5 flex items-center justify-center text-xs mr-3 mt-0.5 flex-shrink-0">{i}</div><span>{t}</span></li>' for i, t in enumerate(summary_list, 1)])
        
        claims_html = ""
        for item in claims_data:
            if "사실" in item['type']: theme = ("text-green-500", "bg-green-50", "border-green-500", "text-green-800", "사실 (Fact)", "fa-check-circle")
            elif "거짓" in item['type']: theme = ("text-red-500", "bg-red-50", "border-red-500", "text-red-800", "거짓/오류 (False)", "fa-times-circle")
            else: theme = ("text-yellow-500", "bg-yellow-50", "border-yellow-500", "text-yellow-800", "의견/해석 (Opinion)", "fa-scale-balanced")
            
            claims_html += f"""
            <div class="flex space-x-4"><div class="mt-1"><i class="fa-solid {theme[5]} {theme[0]} text-xl"></i></div>
            <div><h4 class="font-bold text-gray-900 mb-1">"{item['claim']}"</h4>
            <p class="text-sm text-gray-700 {theme[1]} p-3 rounded-lg border-l-4 {theme[2]}"><strong class="{theme[3]}">{theme[4]}</strong><br>{item['reason']}</p></div></div>"""

        score_theme = ("text-red-600", "border-red-400", "bg-red-50", "매우 자극적") if score >= 70 else \
                      ("text-orange-600", "border-orange-400", "bg-orange-50", "다소 편향됨") if score >= 40 else \
                      ("text-green-600", "border-green-400", "bg-green-50", "중립적/건조함")

        final_html = f"""
        <!DOCTYPE html><html lang="ko"><head><script src="https://cdn.tailwindcss.com"></script><link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet"><style>@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');body {{ font-family: 'Noto Sans KR', sans-serif; background-color: transparent; }}.card {{ background: #ffffff; border-radius: 16px; padding: 24px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); border: 1px solid #f3f4f6; margin-bottom: 20px; }}</style></head><body>
        <div class="max-w-4xl mx-auto space-y-6">
            <div class="card flex items-start space-x-4"><div class="w-24 h-24 bg-gray-100 rounded-xl flex items-center justify-center text-4xl text-gray-400"><i class="fa-regular fa-newspaper"></i></div><div class="flex-1"><h2 class="text-xl font-bold text-gray-900 leading-tight mb-2">{title}</h2><p class="text-sm text-gray-500 mb-2"><i class="fa-solid fa-link mr-1 text-indigo-600"></i> {domain}</p><a href="{url}" target="_blank" class="text-sm text-blue-600 hover:underline font-medium">원본 기사 읽기 <i class="fa-solid fa-external-link-alt text-xs ml-1"></i></a></div></div>
            <div class="card"><h3 class="text-lg font-bold text-gray-900 mb-4 flex items-center"><i class="fa-solid fa-list-check mr-2 text-indigo-600"></i> 핵심 3줄 요약</h3><ul class="space-y-3 text-gray-700">{summary_html}</ul></div>
            <div class="card"><h3 class="text-lg font-bold text-gray-900 mb-6 flex items-center"><i class="fa-solid fa-chart-line mr-2 text-indigo-600"></i> 편향성 & 프레이밍 분석</h3><div class="grid grid-cols-1 md:grid-cols-2 gap-8"><div class="flex flex-col items-center justify-center p-4 bg-gray-50 rounded-xl"><p class="text-sm font-medium text-gray-500 mb-3">🔥 기사 자극성 지수</p><div class="relative w-24 h-24 flex items-center justify-center rounded-full border-8 {score_theme[1]} {score_theme[2]} mb-2"><span class="text-3xl font-bold {score_theme[0]}">{score}</span></div><p class="font-bold {score_theme[0]}">{score_theme[3]}</p></div><div class="flex flex-col justify-center"><div class="mb-4"><p class="text-sm font-medium text-gray-500 mb-1">🧐 프레이밍(의도) 분석</p><p class="text-gray-800 font-semibold text-lg">{framing}</p></div><div class="bg-indigo-50 p-4 rounded-lg border border-indigo-100"><p class="text-xs font-bold text-indigo-800 mb-1"><i class="fa-solid fa-lightbulb"></i> Missing Context</p><p class="text-sm text-indigo-900 leading-relaxed">{comment}</p></div></div></div></div>
            <div class="card"><h3 class="text-lg font-bold text-gray-900 mb-6 flex items-center"><i class="fa-solid fa-magnifying-glass mr-2 text-indigo-600"></i> 팩트체크 리포트</h3><div class="space-y-6">{claims_html}</div></div>
        </div></body></html>"""
        st.markdown(final_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# 🚀 메인 실행부
# ---------------------------------------------------------
st.markdown('<div class="main-title">Veritas Lens <span style="font-size:1.5rem; color:#3B82F6;">Beta</span></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">See the truth behind the noise. URL 하나로 팩트와 편향성을 꿰뚫어 보세요.</div>', unsafe_allow_html=True)

# [수정됨] 에러 방지를 위해 단순화된 텍스트 입력창
url_input = st.text_input("🔗 분석하고 싶은 링크를 입력하세요", placeholder="YouTube 또는 뉴스 기사 URL")

if st.button("Analyze Link 🚀"):
    if not url_input:
        st.warning("링크를 입력해주세요!")
    elif not openai_api_key or not tavily_api_key:
        st.error("기본 API Key(OpenAI, Tavily) 설정이 필요합니다 (사이드바 확인).")
    else:
        llm_instance = get_llm(openai_api_key)
        search_tool = get_search_tool(tavily_api_key)
        
        if any(x in url_input for x in ["youtube.com", "youtu.be", "shorts"]):
            if "RAPIDAPI_KEY" in st.secrets:
                rapid_key = st.secrets["RAPIDAPI_KEY"]
            else:
                rapid_key = rapid_api_key 
            
            if not rapid_key:
                st.error("YouTube 분석을 위해 RapidAPI Key가 필요합니다.")
            else:
                analyze_youtube(url_input, llm_instance, search_tool, rapid_key)
        else:
            analyze_article(url_input, llm_instance, search_tool)

