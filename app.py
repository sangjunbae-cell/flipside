import streamlit as st
import requests
import re
import textwrap
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

# --- 페이지 설정 ---
st.set_page_config(page_title="Veritas Lens", page_icon="👁️", layout="wide")

# --- CSS 커스텀 (들여쓰기 제거를 위해 dedent 적용) ---
st.markdown(textwrap.dedent("""
    <style>
    .main-title {font-size: 3rem; font-weight: 800; color: #111827; letter-spacing: -0.05rem;}
    .sub-title {font-size: 1.2rem; color: #6B7280; margin-bottom: 2rem;}
    div.stButton > button {
        background-color: #2563EB; color: white; border-radius: 8px; 
        padding: 0.5rem 1rem; font-weight: bold; border: none;
        width: 100%;
        transition: all 0.2s;
    }
    div.stButton > button:hover {background-color: #1D4ED8; transform: scale(1.02);}
    </style>
"""), unsafe_allow_html=True)

# --- 사이드바: API 설정 ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    if "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    else:
        openai_api_key = st.text_input("OpenAI API Key", type="password")

    if "TAVILY_API_KEY" in st.secrets:
        tavily_api_key = st.secrets["TAVILY_API_KEY"]
    else:
        tavily_api_key = st.text_input("Tavily API Key", type="password")
        
    st.markdown("---")
    st.subheader("📺 YouTube Unlocker")
    if "RAPIDAPI_KEY" in st.secrets:
        rapid_api_key = st.secrets["RAPIDAPI_KEY"]
    else:
        rapid_api_key = st.text_input("RapidAPI Key", type="password")
        
    st.info("👁️ **Veritas Lens**는 최신 AI와 검색 기술을 결합하여 콘텐츠의 진실을 탐구합니다.")
    
    if st.button("🔄 새로운 분석 시작하기"):
        st.rerun()

# --- 공통 함수 ---

def get_llm(openai_key):
    return ChatOpenAI(temperature=0, openai_api_key=openai_key, model_name="gpt-4o-mini")

def get_search_tool(tavily_key):
    return TavilySearchResults(tavily_api_key=tavily_key, k=3)

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
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
@st.cache_data(show_spinner=False)
def deep_analyze_with_search(text, _llm, _search_tool):
    with st.spinner("🕵️‍♀️ 1단계: 검증이 필요한 핵심 주장을 선별 중..."):
        extraction_prompt = PromptTemplate.from_template("""
        다음 텍스트에서 사실 검증이 필요한 '가장 핵심적인 주장' 3가지를 추출해줘.
        검색 엔진에 입력할 쿼리 형태로 만들어줘.
        
        [텍스트]
        {text}
        
        [출력 형식]
        - 검색쿼리1
        - 검색쿼리2
        - 검색쿼리3
        """)
        claims_result = _llm.invoke(extraction_prompt.format(text=text[:10000])).content
        queries = [line.replace("-", "").strip() for line in claims_result.split('\n') if line.strip().startswith("-")]

    search_context = ""
    with st.spinner(f"🌐 2단계: 웹에서 팩트 확인 중... ({len(queries)}건)"):
        for query in queries[:3]:
            try:
                search_results = _search_tool.invoke(query)
                evidence = "\n".join([f"- 내용: {res['content'][:200]} (출처: {res['url']})" for res in search_results])
                search_context += f"\n[검색 키워드: {query}]\n{evidence}\n"
            except Exception as e:
                pass

    with st.spinner("🧠 3단계: 근거 자료와 대조하여 통합 리포트 작성 중..."):
        final_prompt = PromptTemplate.from_template("""
        당신은 팩트와 논리를 최우선으로 하는 미디어 분석가입니다.
        [원본 텍스트]와 [검색 증거]를 통합하여 분석 리포트를 작성하세요.
        
        [원본 텍스트]
        {text}
        
        [검색 증거]
        {context}
        
        [요청사항]
        1. 핵심주장 분석: 원본의 핵심 주장 3가지를 뽑고, 각 주장에 대해 바로 검증 결과와 근거를 제시하세요.
           - 출처(Source URL)는 반드시 [검색 증거]에 있는 URL 중 가장 신뢰할 수 있는 것을 하나 골라 적으세요.
           - 근거가 없다면 "출처 없음"이라고 적으세요.
        2. 신뢰도 점수: 0~100점
        3. 화자 성향: 1문장 요약
        4. AI 코멘트: 시청자를 위한 조언
        
        [엄격한 출력 포맷]
        SCORE: 75
        STANCE: (화자의 성향)
        COMMENT: (AI 코멘트)
        ANALYSIS:
        - CLAIM: (핵심 주장 1 - 한 문장 요약)
          VERDICT: [사실/거짓/의견/판단보류]
          REASON: (검증 내용 및 이유)
          SOURCE: (http://... 또는 없음)
        - CLAIM: (핵심 주장 2)
          VERDICT: [사실/거짓/의견/판단보류]
          REASON: (이유)
          SOURCE: (URL)
        - CLAIM: (핵심 주장 3)
          VERDICT: [사실/거짓/의견/판단보류]
          REASON: (이유)
          SOURCE: (URL)
        """)
        return _llm.invoke(final_prompt.format(text=text[:10000], context=search_context)).content

# ---------------------------------------------------------
# 🎨 UI 렌더링 함수 (수정됨: dedent 및 문자열 처리 강화)
# ---------------------------------------------------------
def render_report(meta, result):
    # 파싱 로직
    score = 50
    stance = "분석 불가"
    comment = "정보 없음"
    analysis_data = []
    current_item = {}
    
    lines = result.split('\n')
    for line in lines:
        line = line.strip()
        if not line: continue
        
        if line.startswith("SCORE:"): 
            try: score = int(re.findall(r'\d+', line)[0])
            except: score = 50
        elif line.startswith("STANCE:"): stance = line.replace("STANCE:", "").strip()
        elif line.startswith("COMMENT:"): comment = line.replace("COMMENT:", "").strip()
        elif line.startswith("- CLAIM:"):
            if current_item: analysis_data.append(current_item)
            current_item = {"claim": line.replace("- CLAIM:", "").strip()}
        elif line.startswith("VERDICT:"): current_item["verdict"] = line.replace("VERDICT:", "").strip()
        elif line.startswith("REASON:"): current_item["reason"] = line.replace("REASON:", "").strip()
        elif line.startswith("SOURCE:"): current_item["source"] = line.replace("SOURCE:", "").strip()
    
    if current_item: analysis_data.append(current_item)

    # 스타일 설정
    if score >= 70:
        score_theme = ("text-green-600", "border-green-400", "bg-green-50", "신뢰도 높음")
    elif score >= 40:
        score_theme = ("text-yellow-700", "border-yellow-400", "bg-yellow-50", "주의 필요")
    else:
        score_theme = ("text-red-600", "border-red-400", "bg-red-50", "신뢰도 낮음")

    # 분석 카드 HTML 조립 (리스트 컴프리헨션으로 공백 제거)
    cards_html = []
    for idx, item in enumerate(analysis_data, 1):
        verdict = item.get('verdict', '판단보류')
        if "사실" in verdict or "True" in verdict:
            badge = '<span class="px-2 py-1 bg-green-100 text-green-800 text-xs font-bold rounded">✅ 사실 (Fact)</span>'
            border_color = "border-green-200"
        elif "거짓" in verdict or "False" in verdict:
            badge = '<span class="px-2 py-1 bg-red-100 text-red-800 text-xs font-bold rounded">❌ 거짓/오류 (False)</span>'
            border_color = "border-red-200"
        elif "의견" in verdict or "Opinion" in verdict:
            badge = '<span class="px-2 py-1 bg-yellow-100 text-yellow-800 text-xs font-bold rounded">⚠️ 의견 (Opinion)</span>'
            border_color = "border-yellow-200"
        else:
            badge = '<span class="px-2 py-1 bg-gray-100 text-gray-800 text-xs font-bold rounded">❓ 판단보류</span>'
            border_color = "border-gray-200"

        source_url = item.get('source', '없음')
        source_html = ""
        if source_url and "http" in source_url:
            source_html = f"""<div class="mt-3 pt-2 border-t border-dashed border-gray-200"><a href="{source_url}" target="_blank" class="inline-flex items-center text-xs text-blue-600 hover:text-blue-800 transition-colors"><i class="fa-solid fa-link mr-1.5"></i> 검증 출처 보기 (Source)</a></div>"""

        # 카드 HTML 한 줄로 만들기 (들여쓰기 이슈 방지)
        card = f"""<div class="bg-white rounded-xl border {border_color} p-5 shadow-sm hover:shadow-md transition-shadow duration-300"><div class="flex justify-between items-start mb-2"><div class="flex items-center space-x-2"><span class="flex items-center justify-center w-6 h-6 rounded-full bg-blue-100 text-blue-600 text-xs font-bold">{idx}</span><h4 class="font-bold text-gray-900 text-lg">{item.get('claim', '')}</h4></div><div class="flex-shrink-0 ml-2">{badge}</div></div><p class="text-gray-700 text-sm leading-relaxed pl-8 mb-1">{item.get('reason', '')}</p><div class="pl-8">{source_html}</div></div>"""
        cards_html.append(card)

    analysis_section = "".join(cards_html)

    # ⚠️ 중요: HTML 문자열 생성 시 textwrap.dedent를 사용하여
    # 맨 앞의 불필요한 공백을 완전히 제거합니다.
    final_html = textwrap.dedent(f"""
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <div style="font-family: 'Noto Sans KR', sans-serif; max-width: 56rem; margin: 0 auto; padding-top: 1rem;">
        
        <div class="bg-white rounded-2xl p-6 shadow-sm border border-gray-100 flex items-start space-x-5 mb-6">
            <div class="flex-shrink-0">
                <img src="{meta['thumbnail']}" class="w-28 h-28 rounded-xl object-cover shadow-md">
            </div>
            <div class="flex-1">
                <div class="flex items-center space-x-2 mb-1">
                    <span class="px-2 py-0.5 bg-gray-100 text-gray-600 text-xs rounded-full font-medium">{meta['author']}</span>
                </div>
                <h2 class="text-xl font-bold text-gray-900 leading-tight mb-2">{meta['title']}</h2>
                <a href="{meta['url']}" target="_blank" class="inline-flex items-center text-sm text-blue-600 font-medium hover:underline">
                    원본 콘텐츠 확인하기 <i class="fa-solid fa-arrow-up-right-from-square ml-1 text-xs"></i>
                </a>
            </div>
            <div class="flex flex-col items-center justify-center pl-4 border-l border-gray-100">
                <span class="text-xs text-gray-400 font-medium uppercase tracking-wider mb-1">Trust Score</span>
                <div class="relative flex items-center justify-center">
                    <svg class="w-20 h-20 transform -rotate-90">
                        <circle cx="40" cy="40" r="36" stroke="currentColor" stroke-width="8" fill="transparent" class="text-gray-100" />
                        <circle cx="40" cy="40" r="36" stroke="currentColor" stroke-width="8" fill="transparent" class="{score_theme[0]}" stroke-dasharray="{score * 2.26} 226" />
                    </svg>
                    <span class="absolute text-2xl font-bold {score_theme[0]}">{score}</span>
                </div>
                <span class="mt-1 text-xs font-bold {score_theme[0]}">{score_theme[3]}</span>
            </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div class="md:col-span-2 bg-blue-50 rounded-xl p-5 border border-blue-100 relative overflow-hidden">
                <div class="absolute top-0 right-0 p-4 opacity-10"><i class="fa-solid fa-robot text-6xl text-blue-900"></i></div>
                <h3 class="font-bold text-blue-900 mb-2 flex items-center"><i class="fa-solid fa-circle-info mr-2"></i> AI 분석 코멘트</h3>
                <p class="text-sm text-blue-800 leading-relaxed relative z-10">{comment}</p>
            </div>
            <div class="bg-gray-50 rounded-xl p-5 border border-gray-100">
                <h3 class="font-bold text-gray-700 mb-2 text-sm uppercase tracking-wide">화자/논조 성향</h3>
                <p class="text-gray-900 font-medium text-lg leading-tight">{stance}</p>
            </div>
        </div>

        <div class="space-y-4">
            <div class="flex items-center space-x-2 mb-2 px-1">
                <i class="fa-solid fa-magnifying-glass-chart text-blue-600 text-xl"></i>
                <h3 class="text-xl font-bold text-gray-900">핵심 주장 검증 리포트</h3>
            </div>
            {analysis_section}
        </div>

        <div class="text-center pt-8 pb-4">
            <p class="text-xs text-gray-400">Powered by Veritas Lens AI • Tavily Search API</p>
        </div>
    </div>
    """)
    
    st.markdown(final_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# 🚀 메인 실행부
# ---------------------------------------------------------
st.markdown('<div class="main-title">Veritas Lens <span style="font-size:1.5rem; color:#3B82F6;">Beta</span></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">See the truth behind the noise. URL 하나로 팩트와 편향성을 꿰뚫어 보세요.</div>', unsafe_allow_html=True)

# Form을 사용하여 엔터키 입력 지원 및 명시적 제출
with st.form("analyze_form"):
    col1, col2 = st.columns([4, 1])
    with col1:
        url_input = st.text_input("URL 입력", placeholder="YouTube 또는 뉴스 기사 URL을 붙여넣으세요", label_visibility="collapsed")
    with col2:
        submit_btn = st.form_submit_button("Analyze 🚀")

if submit_btn and url_input:
    if not openai_api_key or not tavily_api_key:
        st.error("⚠️ 사이드바에서 API Key를 먼저 설정해주세요.")
    else:
        llm_instance = get_llm(openai_api_key)
        search_tool = get_search_tool(tavily_api_key)
        
        # URL 타입 감지
        if any(x in url_input for x in ["youtube.com", "youtu.be", "shorts"]):
            if "RAPIDAPI_KEY" in st.secrets:
                rapid_key = st.secrets["RAPIDAPI_KEY"]
            else:
                rapid_key = rapid_api_key 
            
            if not rapid_key:
                st.error("YouTube 분석을 위해 RapidAPI Key가 필요합니다.")
            else:
                meta = get_youtube_metadata(url_input)
                with st.spinner("🎧 영상 데이터를 분석 중입니다..."):
                    try:
                        transcript = get_transcript_via_api(url_input, rapid_key)
                        result = deep_analyze_with_search(transcript, llm_instance, search_tool)
                        render_report(meta, result)
                    except Exception as e:
                        st.error(f"오류 발생: {e}")
        else:
            # 뉴스 분석
            try:
                with st.spinner("📰 기사 내용을 분석 중입니다..."):
                    loader = WebBaseLoader(url_input)
                    docs = loader.load()
                    content = docs[0].page_content[:15000]
                    title = docs[0].metadata.get('title', '뉴스 기사')
                    domain = url_input.split("//")[-1].split("/")[0].replace("www.", "")
                    meta = {"title": title, "author": domain, "thumbnail": "https://cdn-icons-png.flaticon.com/512/2965/2965879.png", "url": url_input}
                    
                    result = deep_analyze_with_search(content, llm_instance, search_tool)
                    render_report(meta, result)
            except Exception as e:
                st.error(f"오류 발생: {e}")
