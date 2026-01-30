import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
import re

# --- 페이지 설정 ---
st.set_page_config(page_title="InsightLens AI", page_icon="🕵️", layout="wide")

# --- CSS 커스텀 (UI 폴리싱) ---
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1E3A8A;}
    .sub-header {font-size: 1.5rem; font-weight: 600; color: #4B5563;}
    .card {background-color: #f9fafb; padding: 20px; border-radius: 10px; border: 1px solid #e5e7eb; margin-bottom: 20px;}
    .fact-box {padding: 15px; border-radius: 8px; margin-bottom: 10px;}
    .fact-true {background-color: #ecfdf5; border-left: 5px solid #10b981;}
    .fact-check {background-color: #fff7ed; border-left: 5px solid #f97316;}
    </style>
""", unsafe_allow_html=True)

# --- 사이드바: 모드 선택 & 설정 ---
with st.sidebar:
    st.title("🕵️ InsightLens AI")
    st.markdown("---")
    mode = st.radio("분석 모드 선택", ["🎥 유튜브 영상 분석", "📰 뉴스 기사 분석"])
    st.markdown("---")
    
    # Secrets에서 키를 먼저 찾아보고, 없으면 입력창을 띄웁니다
    if "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    else:
        openai_api_key = st.text_input("OpenAI API Key", type="password")

    if "TAVILY_API_KEY" in st.secrets:
        tavily_api_key = st.secrets["TAVILY_API_KEY"]
    else:
        tavily_api_key = st.text_input("Tavily API Key", type="password")
        
    st.info("💡 이 툴은 AI와 실시간 검색(RAG)을 결합하여 콘텐츠의 편향성과 사실 여부를 검증합니다.")

# --- 공통 함수 ---
def get_llm(openai_key):
    return ChatOpenAI(temperature=0, openai_api_key=openai_key, model_name="gpt-4o")

def get_search_tool(tavily_key):
    return TavilySearchResults(tavily_api_key=tavily_key, k=3)

# ---------------------------------------------------------
# 🎥 모드 1: 유튜브 분석 로직
# ---------------------------------------------------------
def run_youtube_analysis():
    st.markdown('<div class="main-header">🎥 유튜브 팩트체커</div>', unsafe_allow_html=True)
    
    url = st.text_input("유튜브 링크를 입력하세요", placeholder="https://youtu.be/...")
    
    if st.button("영상 분석 시작"):
        if not openai_api_key or not tavily_api_key:
            st.error("API Key를 먼저 입력해주세요.")
            return
            
        # 1. 자막 추출
        video_id = None
        if "v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be" in url:
            video_id = url.split("/")[-1]
            
        if not video_id:
            st.error("올바르지 않은 유튜브 링크입니다.")
            return

        try:
            with st.spinner("자막을 다운로드 중입니다..."):
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
                full_text = " ".join([t['text'] for t in transcript_list])[:4000] # 비용 절감용 길이 제한
        except Exception as e:
            st.error(f"자막을 가져올 수 없습니다. (에러: {e})")
            return

        # 2. AI 분석 (주장 추출 -> 검색 -> 검증)
        try:
            llm = get_llm(openai_api_key)
            search = get_search_tool(tavily_api_key)
            
            with st.spinner("🕵️‍♀️ AI가 영상을 시청하고 팩트체크를 진행 중입니다..."):
                # A. 핵심 주장 추출
                claims_prompt = PromptTemplate.from_template("다음 텍스트에서 팩트체크가 필요한 핵심 주장 3가지만 요약해줘:\n{text}")
                claims = llm.invoke(claims_prompt.format(text=full_text)).content
                
                # B. 결과 리포트 생성
                st.markdown("### 📊 분석 리포트")
                
                tab1, tab2 = st.tabs(["팩트체크 결과", "영상 요약"])
                
                with tab1:
                    # 간단한 파싱 후 루프
                    lines = [line for line in claims.split('\n') if line.strip()]
                    for line in lines:
                        if len(line) < 5: continue
                        
                        # 검색 실행
                        try:
                            search_result = search.invoke(line)
                            evidence = str(search_result)
                        except Exception:
                            evidence = "검색 실패"
                        
                        # 최종 검증
                        verify_prompt = PromptTemplate.from_template(
                            "주장: {claim}\n증거: {evidence}\n위 증거를 바탕으로 주장이 '사실', '거짓', '판단보류' 중 무엇인지, 그리고 그 이유를 한 문장으로 써줘."
                        )
                        verdict = llm.invoke(verify_prompt.format(claim=line, evidence=evidence)).content
                        
                        # UI 출력
                        color_class = "fact-true" if "사실" in verdict else "fact-check"
                        st.markdown(f"""
                            <div class='fact-box {color_class}'>
                                <strong>🗣️ 주장:</strong> {line}<br>
                                <strong>🤖 AI 판정:</strong> {verdict}
                            </div>
                        """, unsafe_allow_html=True)
                
                with tab2:
                    summary = llm.invoke(f"다음 내용을 3줄로 요약해줘:\n{full_text}").content
                    st.info(summary)
        except Exception as e:
             st.error(f"분석 중 오류가 발생했습니다: {e}")

# ---------------------------------------------------------
# 📰 모드 2: 뉴스 기사 분석 로직
# ---------------------------------------------------------
def run_news_analysis():
    st.markdown('<div class="main-header">📰 뉴스 딥다이브 & 편향성 분석</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        headline = st.text_input("기사 제목 (Headline)")
    with col2:
        media_name = st.text_input("언론사명 (선택사항)")
        
    article_body = st.text_area("기사 본문 내용 (여기에 붙여넣으세요)", height=200)
    
    if st.button("기사 분석 시작"):
        if not openai_api_key or not article_body:
            st.error("API Key와 본문 내용은 필수입니다.")
            return

        try:
            llm = get_llm(openai_api_key)
            
            with st.spinner("🔍 기사의 행간을 읽고 누락된 맥락을 찾는 중입니다..."):
                
                # 1. 자극성 & 프레이밍 분석
                bias_prompt = PromptTemplate.from_template("""
                    기사 제목: {headline}
                    본문: {body}
                    
                    이 기사를 분석해서 다음 2가지를 알려줘:
                    1. 자극성 점수 (0~100점)와 그 이유
                    2. 이 기사가 독자에게 심어주려는 프레임(의도)
                """)
                bias_result = llm.invoke(bias_prompt.format(headline=headline, body=article_body)).content
                
                # 2. 누락된 맥락 검색 (RAG)
                search = get_search_tool(tavily_api_key)
                context_query = llm.invoke(f"이 기사 '{headline}'의 주장을 반박하거나 보완하기 위해 검색해야 할 키워드 1개만 알려줘.").content
                search_res = search.invoke(context_query)
                
                missing_context = llm.invoke(f"""
                    기사 내용: {article_body}
                    검색된 외부 사실: {search_res}
                    
                    위 '검색된 외부 사실'에는 있지만, '기사 내용'에서는 쏙 빠져있는(누락된) 중요한 맥락 1가지만 찾아서 설명해줘.
                """).content

                # --- 결과 출력 ---
                st.markdown("### ⚖️ 분석 결과")
                
                # 자극성 게이지 (텍스트로 표현)
                st.markdown(f"<div class='card'>{bias_result}</div>", unsafe_allow_html=True)
                
                st.markdown("### 🧩 누락된 퍼즐 조각 (Missing Context)")
                st.markdown(f"""
                    <div class='fact-box fact-check'>
                        <strong>⚠️ AI가 찾은 빠진 맥락:</strong><br>
                        {missing_context}
                    </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {e}")

# --- 메인 실행 ---
if mode == "🎥 유튜브 영상 분석":
    run_youtube_analysis()
else:
    run_news_analysis()
