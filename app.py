import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
import re

# --- 페이지 설정 ---
st.set_page_config(page_title="Veritas Lens", page_icon="👁️", layout="wide")

# --- CSS 커스텀 (모던 & 섹시한 UI) ---
st.markdown("""
    <style>
    .main-title {font-size: 3rem; font-weight: 800; color: #111827; letter-spacing: -0.05rem;}
    .sub-title {font-size: 1.2rem; color: #6B7280; margin-bottom: 2rem;}
    .card {background-color: #ffffff; padding: 25px; border-radius: 15px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 20px;}
    .fact-box {padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left-width: 5px;}
    .fact-true {background-color: #ecfdf5; border-color: #10b981; color: #065f46;}
    .fact-check {background-color: #fff7ed; border-color: #f97316; color: #9a3412;}
    .bias-gauge {font-size: 1.5rem; font-weight: bold; text-align: center; margin: 10px 0;}
    </style>
""", unsafe_allow_html=True)

# --- 사이드바: API 설정 ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Secrets 관리
    if "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    else:
        openai_api_key = st.text_input("OpenAI API Key", type="password")

    if "TAVILY_API_KEY" in st.secrets:
        tavily_api_key = st.secrets["TAVILY_API_KEY"]
    else:
        tavily_api_key = st.text_input("Tavily API Key", type="password")
        
    st.markdown("---")
    st.info("👁️ **Veritas Lens**는 콘텐츠의 URL만 입력하면 자동으로 형식을 감지하여 편향성과 사실 관계를 분석합니다.")

# --- 공통 함수 ---
def get_llm(openai_key):
    return ChatOpenAI(temperature=0, openai_api_key=openai_key, model_name="gpt-4o")

def get_search_tool(tavily_key):
    return TavilySearchResults(tavily_api_key=tavily_key, k=3)

# ---------------------------------------------------------
# 🧠 분석 로직 1: 유튜브 (Youtube Analysis)
# ---------------------------------------------------------
def analyze_youtube(url, llm, search):
    # 1. Video ID 추출
    video_id = None
    if "v=" in url:
        video_id = url.split("v=")[1].split("&")[0]
    elif "youtu.be" in url:
        video_id = url.split("/")[-1]
    elif "shorts" in url:
         video_id = url.split("shorts/")[1].split("?")[0]
         
    if not video_id:
        st.error("유효하지 않은 유튜브 링크입니다.")
        return

    # 2. 자막 추출
    try:
        with st.spinner("🎧 영상의 자막을 추출하고 있습니다..."):
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
            full_text = " ".join([t['text'] for t in transcript_list])[:6000] # 길이 제한 상향
    except Exception as e:
        st.error(f"자막을 가져올 수 없습니다. (에러: {e})")
        return

    # 3. 분석 시작
    with st.spinner("👁️ Veritas Lens가 영상을 분석 중입니다..."):
        # 요약 및 주장 추출
        analysis_prompt = PromptTemplate.from_template("""
        다음 유튜브 스크립트를 분석해줘:
        
        [스크립트]
        {text}
        
        [요청사항]
        1. 이 영상의 핵심 주제를 3줄로 요약해줘.
        2. 팩트체크가 필요한 구체적인 주장(Fact Claims) 3가지만 추출해줘.
        
        형식:
        SUMMARY: ...
        CLAIMS:
        - 주장1
        - 주장2
        - 주장3
        """)
        
        analysis_result = llm.invoke(analysis_prompt.format(text=full_text)).content
        
        # 결과 파싱
        summary_text = ""
        claims_list = []
        
        if "SUMMARY:" in analysis_result and "CLAIMS:" in analysis_result:
            parts = analysis_result.split("CLAIMS:")
            summary_text = parts[0].replace("SUMMARY:", "").strip()
            claims_list = [c.strip("- ").strip() for c in parts[1].split("\n") if c.strip()]
        else:
            summary_text = analysis_result # 파싱 실패 시 전체 출력
            
        # UI 출력
        st.markdown(f"<div class='card'><h3>📺 영상 요약</h3>{summary_text}</div>", unsafe_allow_html=True)
        
        st.markdown("### 🕵️ 팩트체크 리포트")
        for claim in claims_list:
            if len(claim) < 5: continue
            
            # RAG 검색
            try:
                search_res = search.invoke(claim)
                evidence = str(search_res)
            except:
                evidence = "검색 결과 없음"
            
            # 검증
            verify_prompt = PromptTemplate.from_template(
                "주장: {claim}\n증거: {evidence}\n증거를 바탕으로 이 주장이 '사실', '거짓', '판단보류' 중 무엇인지 판단하고 이유를 1문장으로 써줘."
            )
            verdict = llm.invoke(verify_prompt.format(claim=claim, evidence=evidence)).content
            
            color_class = "fact-true" if "사실" in verdict else "fact-check"
            st.markdown(f"<div class='fact-box {color_class}'><strong>🗣️ {claim}</strong><br>↳ {verdict}</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🧠 분석 로직 2: 웹 뉴스 (Web Article Analysis)
# ---------------------------------------------------------
def analyze_article(url, llm, search):
    try:
        with st.spinner("📰 기사 본문을 읽어오는 중입니다..."):
            loader = WebBaseLoader(url)
            docs = loader.load()
            article_content = docs[0].page_content[:6000] # 길이 제한
            article_title = docs[0].metadata.get('title', '제목 없음')
    except Exception as e:
        st.error(f"기사를 읽어올 수 없습니다. (접근 차단 또는 잘못된 URL): {e}")
        return

    with st.spinner("⚖️ 기사의 편향성과 맥락을 분석 중입니다..."):
        # 편향성 및 맥락 분석
        bias_prompt = PromptTemplate.from_template("""
        기사 제목: {title}
        기사 본문: {text}
        
        다음 3가지를 분석해줘:
        1. 자극성 점수 (0~100점)
        2. 이 기사의 프레이밍(의도) 요약
        3. 이 기사의 주장을 검증하기 위해 검색해야 할 키워드 1개
        
        형식:
        SCORE: 80
        FRAMING: ...
        KEYWORD: ...
        """)
        
        bias_res = llm.invoke(bias_prompt.format(title=article_title, text=article_content)).content
        
        # 파싱
        score = "N/A"
        framing = "분석 실패"
        keyword = article_title
        
        for line in bias_res.split('\n'):
            if "SCORE:" in line: score = line.split(":")[1].strip()
            if "FRAMING:" in line: framing = line.split(":")[1].strip()
            if "KEYWORD:" in line: keyword = line.split(":")[1].strip()
            
        # RAG 검색 (누락된 맥락 찾기)
        search_res = search.invoke(keyword)
        missing_context = llm.invoke(f"기사 내용: {article_content}\n외부 사실: {search_res}\n기사에서 누락된 중요한 맥락 1가지만 찾아서 설명해줘.").content
        
        # UI 출력
        st.markdown(f"<div class='card'><h3>📰 {article_title}</h3></div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
             st.markdown(f"<div class='card'><div class='bias-gauge'>🔥 자극성 지수: {score}</div></div>", unsafe_allow_html=True)
        with col2:
             st.markdown(f"<div class='card'><strong>🔍 프레이밍:</strong><br>{framing}</div>", unsafe_allow_html=True)
             
        st.markdown(f"<div class='fact-box fact-check'><strong>🧩 놓친 맥락(Missing Context):</strong><br>{missing_context}</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🚀 메인 실행부 (Router)
# ---------------------------------------------------------
st.markdown('<div class="main-title">Veritas Lens <span style="font-size:1.5rem; color:#3B82F6;">Beta</span></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">See the truth behind the noise. URL 하나로 팩트와 편향성을 꿰뚫어 보세요.</div>', unsafe_allow_html=True)

url_input = st.text_input("🔗 분석하고 싶은 링크를 입력하세요 (YouTube or News URL)", placeholder="https://...")

if st.button("Analyze Link 🚀"):
    if not url_input:
        st.warning("링크를 입력해주세요!")
    elif not openai_api_key or not tavily_api_key:
        st.error("API Key 설정이 필요합니다 (사이드바 확인).")
    else:
        llm_instance = get_llm(openai_api_key)
        search_tool = get_search_tool(tavily_api_key)
        
        # --- 자동 감지 로직 (Auto-Router) ---
        if "youtube.com" in url_input or "youtu.be" in url_input:
            analyze_youtube(url_input, llm_instance, search_tool)
        else:
            analyze_article(url_input, llm_instance, search_tool)
