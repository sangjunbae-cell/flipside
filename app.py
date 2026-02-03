import streamlit as st
import requests
import re
import textwrap  # 👈 [필수 추가] 들여쓰기 제거용
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
    return ChatOpenAI(temperature=0, openai_api_key=openai_key, model_name="gpt-4o-mini")

def get_search_tool(tavily_key):
    return TavilySearchResults(tavily_api_key=tavily_key, k=3)

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

def get_transcript_via_api(video_url, api_key):
    url = "https://youtube-transcript3.p.rapidapi.com/api/transcript-with-url"
    querystring = {"url": video_url, "flat_text": "true", "lang": "ko"}
    headers = {"x-rapidapi-key": api_key, "x-rapidapi-host": "youtube-transcript3.p.rapidapi.com"}

    try:
        response = requests.get(url, headers=headers, params=querystring, timeout=20)
        
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

# --- RAG 심층 분석 파이프라인 ---
def deep_analyze_with_search(text, llm, search_tool):
    # 1단계: 주장 추출
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

    # 2단계: 웹 검색
    search_context = ""
    with st.spinner(f"🌐 2단계: 웹에서 팩트 확인 중... ({len(queries)}건)"):
        for query in queries[:3]:
            try:
                search_results = search_tool.invoke(query)
                evidence = "\n".join([f"- 출처({res['url']}): {res['content'][:200]}" for res in search_results])
                search_context += f"\n[검색 키워드: {query}]\n{evidence}\n"
            except Exception as e:
                pass

    # 3단계: 종합 분석
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
        3. 화자성향: 1문장 요약
        4. AI코멘트: 1문장
        5. 팩트체크: 3가지 판별 [사실/거짓/의견]
        
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
# 🎨 [UX 1] 유튜브 분석 함수 (수정됨: dedent 적용)
# ---------------------------------------------------------
def analyze_youtube(url, llm, search, api_key):
    meta = get_youtube_metadata(url)
    full_text = ""
    
    with st.spinner("🎧 영상 데이터를 가져오는 중... (RapidAPI)"):
        try:
            full_text = get_transcript_via_api(url, api_key)
        except Exception as e:
            st.error(f"❌ 영상 자막을 가져올 수 없습니다: {e}")
            return

    try:
        result = deep_analyze_with_search(full_text, llm, search)
    except Exception as e:
        st.error(f"AI 분석 오류: {e}")
        return

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
        
        if current_section == "SUMMARY" and line.startswith("-"): 
            summary_list.append(line.replace("-", "").strip())
        if current_section == "CLAIMS" and line.startswith("-"):
            parts = line.replace("-", "").strip().split("|")
            if len(parts) >= 3: 
                claims_data.append({
                    "claim": parts[0].strip(), 
                    "type": parts[1].strip(), 
                    "reason": parts[2].strip()
                })

    # 스타일 설정
    if score >= 70:
        score_msg, score_sub = "신뢰도 높음 (Trustworthy)", "팩트와 근거가 충분합니다."
        gauge_bg, gauge_border, text_color = "bg-green-50", "border-green-500", "text-green-700"
    elif score >= 40:
        score_msg, score_sub = "주의 필요 (Caution)", "팩트와 주관적 견해가 섞여 있습니다."
        gauge_bg, gauge_border, text_color = "bg-yellow-50", "border-yellow-400", "text-yellow-700"
    else:
        score_msg, score_sub = "신뢰도 낮음 (Low Trust)", "검증되지 않은 주장이 많습니다."
        gauge_bg, gauge_border, text_color = "bg-red-50", "border-red-500", "text-red-700"

    # HTML 조각 생성 (textwrap.dedent로 들여쓰기 제거 중요!)
    summary_html = ""
    for i, item in enumerate(summary_list, 1):
        summary_html += f"""
        <li class="flex items-start">
            <i class="fa-solid fa-{i} text-blue-100 bg-blue-600 rounded-full w-5 h-5 flex items-center justify-center text-xs mr-3 mt-0.5"></i>
            <span>{item}</span>
        </li>"""

    claims_html = ""
    for item in claims_data:
        if "사실" in item['type']: icon, bg_cls, lbl = "fa-check-circle text-green-500", "bg-green-50 border-green-500 text-green-800", "사실 (Fact)"
        elif "거짓" in item['type']: icon, bg_cls, lbl = "fa-times-circle text-red-500", "bg-red-50 border-red-500 text-red-800", "거짓/오류 (False)"
        else: icon, bg_cls, lbl = "fa-triangle-exclamation text-yellow-500", "bg-yellow-50 border-yellow-500 text-yellow-800", "의견/전망 (Opinion)"

        claims_html += f"""
        <div class="flex space-x-4">
            <div class="mt-1"><i class="fa-solid {icon} text-xl"></i></div>
            <div>
                <h4 class="font-bold text-gray-900 mb-1">"{item['claim']}"</h4>
                <p class="text-sm text-gray-700 {bg_cls} p-3 rounded-lg border-l-4">
                    <strong>{lbl}</strong><br>{item['reason']}
                </p>
            </div>
        </div>"""

    # --- HTML 조립 (textwrap.dedent 사용) ---
    final_html = textwrap.dedent(f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
            body {{ font-family: 'Noto Sans KR', sans-serif; }}
            .card {{ background: #ffffff; border-radius: 16px; padding: 24px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03); border: 1px solid #f3f4f6; margin-bottom: 1.5rem; }}
        </style>
    </head>
    <body class="bg-transparent text-gray-800">
        <div class="max-w-3xl mx-auto py-4">
            <div class="card flex items-start space-x-4">
                <img src="{meta['thumbnail']}" class="w-24 h-24 rounded-xl object-cover shadow-sm">
                <div>
                    <h2 class="font-bold text-gray-900 leading-tight mb-1" style="font-size: 1.1rem;">{meta['title']}</h2>
                    <p class="text-sm text-gray-500 mb-2"><i class="fa-brands fa-youtube mr-1 text-red-600"></i> {meta['author']}</p>
                    <a href="{meta['url']}" target="_blank" class="text-sm text-blue-600 hover:underline inline-block">원본 영상 보기 <i class="fa-solid fa-external-link-alt text-xs ml-1"></i></a>
                </div>
            </div>
            <div class="card">
                <h3 class="text-lg font-bold text-gray-900 mb-4"><i class="fa-solid fa-list-check mr-2 text-blue-600"></i> 핵심 3줄 요약</h3>
                <ul class="space-y-3 text-gray-700 pl-1">{summary_html}</ul>
            </div>
            <div class="card">
                <h3 class="text-lg font-bold text-gray-900 mb-6"><i class="fa-solid fa-scale-balanced mr-2 text-blue-600"></i> 편향성 및 신뢰도 분석</h3>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div class="text-center md:text-left">
                        <p class="text-sm font-medium text-gray-500 mb-2">AI 신뢰도 점수</p>
                        <div class="flex items-center justify-center md:justify-start space-x-3">
                            <div class="relative w-20 h-20 flex items-center justify-center rounded-full border-4 {gauge_border} {gauge_bg}">
                                <span class="text-2xl font-bold {text_color}">{score}<small class="text-sm font-normal">/100</small></span>
                            </div>
                            <div class="text-left">
                                <p class="font-bold text-gray-900">{score_msg}</p>
                                <p class="text-xs text-gray-500">{score_sub}</p>
                            </div>
                        </div>
                    </div>
                    <div>
                         <p class="text-sm font-medium text-gray-500 mb-3">화자 성향</p>
                         <div class="flex justify-between text-xs text-gray-400 mb-1 px-1"><span>비판적</span><span>옹호적</span></div>
                        <div class="relative h-4 bg-gray-200 rounded-full overflow-hidden">
                            <div class="absolute top-0 left-0 h-full bg-gradient-to-r from-gray-300 to-blue-500" style="width: 60%;"></div>
                        </div>
                        <p class="text-right text-xs text-blue-600 font-medium mt-1">{stance}</p>
                    </div>
                </div>
                <div class="mt-8 p-4 bg-blue-50 rounded-xl border border-blue-100">
                    <h4 class="font-bold text-sm text-blue-900 mb-2"><i class="fa-solid fa-circle-info mr-1"></i> AI의 분석 코멘트</h4>
                    <p class="text-sm text-blue-800 leading-relaxed">{comment}</p>
                </div>
            </div>
            <div class="card">
                 <h3 class="text-lg font-bold text-gray-900 mb-6"><i class="fa-solid fa-magnifying-glass mr-2 text-blue-600"></i> 주요 주장 팩트체크</h3>
                 <div class="space-y-6">{claims_html}</div>
            </div>
            <div class="text-center text-xs text-gray-400 pb-8">※ 본 리포트는 AI와 실시간 웹 검색(RAG)을 통해 자동 생성되었습니다.</div>
        </div>
    </body>
    </html>""")

    st.markdown(final_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# 🎨 [UX 2] 뉴스 분석 함수 (수정됨: dedent 적용)
# ---------------------------------------------------------
def analyze_article(url, llm, search):
    try:
        with st.spinner("📰 기사 본문을 읽어오는 중..."):
            loader = WebBaseLoader(url)
            docs = loader.load()
            content = docs[0].page_content[:15000]
            title = docs[0].metadata.get('title', '뉴스 기사 분석')
            domain = url.split("//")[-1].split("/")[0].replace("www.", "")
    except Exception as e:
        st.error(f"기사를 읽어올 수 없습니다: {e}")
        return

    try:
        result = deep_analyze_with_search(content, llm, search)
    except Exception as e:
        st.error(f"AI 분석 오류: {e}")
        return

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
        
        if current_section == "SUMMARY" and line.startswith("-"): 
            summary_list.append(line.replace("-", "").strip())
        if current_section == "CLAIMS" and line.startswith("-"):
            parts = line.replace("-", "").strip().split("|")
            if len(parts) >= 3: 
                claims_data.append({
                    "claim": parts[0].strip(), 
                    "type": parts[1].strip(), 
                    "reason": parts[2].strip()
                })

    if score >= 70:
        score_msg, score_sub = "신뢰도 높음 (Trustworthy)", "팩트와 근거가 충분합니다."
        gauge_bg, gauge_border, text_color = "bg-green-50", "border-green-500", "text-green-700"
    elif score >= 40:
        score_msg, score_sub = "주의 필요 (Caution)", "팩트와 주관적 견해가 섞여 있습니다."
        gauge_bg, gauge_border, text_color = "bg-yellow-50", "border-yellow-400", "text-yellow-700"
    else:
        score_msg, score_sub = "신뢰도 낮음 (Low Trust)", "검증되지 않은 주장이 많습니다."
        gauge_bg, gauge_border, text_color = "bg-red-50", "border-red-500", "text-red-700"

    summary_html = ""
    for i, item in enumerate(summary_list, 1):
        summary_html += f"""
        <li class="flex items-start">
            <i class="fa-solid fa-{i} text-indigo-100 bg-indigo-600 rounded-full w-5 h-5 flex items-center justify-center text-xs mr-3 mt-0.5"></i>
            <span>{item}</span>
        </li>"""

    claims_html = ""
    for item in claims_data:
        if "사실" in item['type']: icon, bg_cls, lbl = "fa-check-circle text-green-500", "bg-green-50 border-green-500 text-green-800", "사실 (Fact)"
        elif "거짓" in item['type']: icon, bg_cls, lbl = "fa-times-circle text-red-500", "bg-red-50 border-red-500 text-red-800", "거짓/오류 (False)"
        else: icon, bg_cls, lbl = "fa-triangle-exclamation text-yellow-500", "bg-yellow-50 border-yellow-500 text-yellow-800", "의견/전망 (Opinion)"

        claims_html += f"""
        <div class="flex space-x-4">
            <div class="mt-1"><i class="fa-solid {icon} text-xl"></i></div>
            <div>
                <h4 class="font-bold text-gray-900 mb-1">"{item['claim']}"</h4>
                <p class="text-sm text-gray-700 {bg_cls} p-3 rounded-lg border-l-4">
                    <strong>{lbl}</strong><br>{item['reason']}
                </p>
            </div>
        </div>"""

    # --- HTML 조립 (textwrap.dedent 사용) ---
    final_html = textwrap.dedent(f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
            body {{ font-family: 'Noto Sans KR', sans-serif; }}
            .card {{ background: #ffffff; border-radius: 16px; padding: 24px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03); border: 1px solid #f3f4f6; margin-bottom: 1.5rem; }}
        </style>
    </head>
    <body class="bg-transparent text-gray-800">
        <div class="max-w-3xl mx-auto py-4">
            <div class="card flex items-start space-x-4">
                <div class="w-24 h-24 bg-gray-100 rounded-xl flex items-center justify-center text-4xl text-gray-400 flex-shrink-0">
                    <i class="fa-regular fa-newspaper"></i>
                </div>
                <div>
                    <h2 class="font-bold text-gray-900 leading-tight mb-1" style="font-size: 1.1rem;">{title}</h2>
                    <p class="text-sm text-gray-500 mb-2"><i class="fa-solid fa-link mr-1 text-indigo-600"></i> {domain}</p>
                    <a href="{url}" target="_blank" class="text-sm text-blue-600 hover:underline inline-block">원본 기사 읽기 <i class="fa-solid fa-external-link-alt text-xs ml-1"></i></a>
                </div>
            </div>
            <div class="card">
                <h3 class="text-lg font-bold text-gray-900 mb-4"><i class="fa-solid fa-list-check mr-2 text-indigo-600"></i> 핵심 3줄 요약</h3>
                <ul class="space-y-3 text-gray-700 pl-1">{summary_html}</ul>
            </div>
            <div class="card">
                <h3 class="text-lg font-bold text-gray-900 mb-6"><i class="fa-solid fa-scale-balanced mr-2 text-indigo-600"></i> 편향성 및 신뢰도 분석</h3>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div class="text-center md:text-left">
                        <p class="text-sm font-medium text-gray-500 mb-2">AI 신뢰도 점수</p>
                        <div class="flex items-center justify-center md:justify-start space-x-3">
                            <div class="relative w-20 h-20 flex items-center justify-center rounded-full border-4 {gauge_border} {gauge_bg}">
                                <span class="text-2xl font-bold {text_color}">{score}<small class="text-sm font-normal">/100</small></span>
                            </div>
                            <div class="text-left">
                                <p class="font-bold text-gray-900">{score_msg}</p>
                                <p class="text-xs text-gray-500">{score_sub}</p>
                            </div>
                        </div>
                    </div>
                    <div>
                         <p class="text-sm font-medium text-gray-500 mb-3">기사 논조</p>
                         <div class="flex justify-between text-xs text-gray-400 mb-1 px-1"><span>비판적</span><span>옹호적</span></div>
                        <div class="relative h-4 bg-gray-200 rounded-full overflow-hidden">
                            <div class="absolute top-0 left-0 h-full bg-gradient-to-r from-gray-300 to-indigo-500" style="width: 50%;"></div>
                        </div>
                        <p class="text-right text-xs text-indigo-600 font-medium mt-1">{stance}</p>
                    </div>
                </div>
                <div class="mt-8 p-4 bg-indigo-50 rounded-xl border border-indigo-100">
                    <h4 class="font-bold text-sm text-indigo-900 mb-2"><i class="fa-solid fa-circle-info mr-1"></i> AI의 분석 코멘트</h4>
                    <p class="text-sm text-indigo-800 leading-relaxed">{comment}</p>
                </div>
            </div>
            <div class="card">
                 <h3 class="text-lg font-bold text-gray-900 mb-6"><i class="fa-solid fa-magnifying-glass mr-2 text-indigo-600"></i> 주요 주장 팩트체크</h3>
                 <div class="space-y-6">{claims_html}</div>
            </div>
            <div class="text-center text-xs text-gray-400 pb-8">※ 본 리포트는 AI와 실시간 웹 검색(RAG)을 통해 자동 생성되었습니다.</div>
        </div>
    </body>
    </html>""")

    st.markdown(final_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# 🚀 메인 실행부
# ---------------------------------------------------------
st.markdown('<div class="main-title">Veritas Lens <span style="font-size:1.5rem; color:#3B82F6;">Beta</span></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">See the truth behind the noise. URL 하나로 팩트와 편향성을 꿰뚫어 보세요.</div>', unsafe_allow_html=True)

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
