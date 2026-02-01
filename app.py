import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
import requests # API 호출용
import re

# --- 페이지 설정 ---
st.set_page_config(page_title="Veritas Lens", page_icon="👁️", layout="wide")

# --- CSS 커스텀 ---
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
        rapid_api_key = st.text_input("RapidAPI Key (X-RapidAPI-Key)", type="password")
        
    st.info("👁️ **Veritas Lens**는 미들웨어 API를 통해 차단 없이 영상을 분석합니다.")

# --- 공통 함수 ---
def get_llm(openai_key):
    return ChatOpenAI(temperature=0, openai_api_key=openai_key, model_name="gpt-4o-mini")

def get_search_tool(tavily_key):
    return TavilySearchResults(tavily_api_key=tavily_key, k=3)

# 🚀 [NEW] 구독하신 'YouTube Transcript 3' API 맞춤형 함수
def get_transcript_via_api(video_url, api_key):
    # 1. API 엔드포인트 설정 (보내주신 코드 기반)
    url = "https://youtube-transcript3.p.rapidapi.com/api/transcript-with-url"
    
    # 2. 파라미터 설정 (flat_text=true 옵션 사용으로 텍스트만 깔끔하게 받음)
    # 팁: 한국어 분석을 위해 lang='ko'를 우선 시도합니다.
    querystring = {
        "url": video_url,
        "flat_text": "true",
        "lang": "ko" 
    }

    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "youtube-transcript3.p.rapidapi.com"
    }

    # 3. API 호출
    response = requests.get(url, headers=headers, params=querystring)
    
    # 4. 에러 처리 및 언어 Fallback (한국어 없으면 영어로 재시도)
    if response.status_code != 200:
        # 한국어가 없어서 에러가 났을 수도 있으니 영어로 재시도
        querystring["lang"] = "en"
        response = requests.get(url, headers=headers, params=querystring)
        
        if response.status_code != 200:
            raise Exception(f"API 호출 실패 ({response.status_code}): {response.text}")

    # 5. 데이터 파싱
    data = response.json()
    
    # flat_text=true 옵션을 썼으므로, 보통 'transcript' 키에 통문장이 들어옵니다.
    if "transcript" in data:
        return data["transcript"][:15000] # 길이 제한
    else:
        # 만약 구조가 다르면 통째로 반환 (디버깅용)
        return str(data)[:15000]

# ---------------------------------------------------------
# 🧠 분석 로직 1: 유튜브 (RapidAPI 사용)
# ---------------------------------------------------------
def analyze_youtube(url, llm, search, rapid_key):
    
    # 1. 자막 추출 (미들웨어 사용)
    full_text = ""
    with st.spinner("🚀 RapidAPI를 통해 자막을 가져오고 있습니다..."):
        try:
            full_text = get_transcript_via_api(url, rapid_key)
        except Exception as e:
            st.error(f"❌ 자막 가져오기 실패: {e}")
            st.warning("RapidAPI Key가 정확한지 확인해주세요.")
            return

    # 2. 분석 시작
    with st.spinner("👁️ Veritas Lens가 내용을 분석 중입니다..."):
        analysis_prompt = PromptTemplate.from_template("""
        다음 텍스트를 분석해줘:
        {text}
        
        [요청사항]
        1. 핵심 주제를 3줄로 요약해줘.
        2. 팩트체크가 필요한 구체적인 주장(Fact Claims) 3가지만 추출해줘.
        
        형식:
        SUMMARY: ...
        CLAIMS:
        - 주장1
        - 주장2
        - 주장3
        """)
        
        analysis_result = llm.invoke(analysis_prompt.format(text=full_text)).content
        
        summary_text = ""
        claims_list = []
        
        if "SUMMARY:" in analysis_result and "CLAIMS:" in analysis_result:
            parts = analysis_result.split("CLAIMS:")
            summary_text = parts[0].replace("SUMMARY:", "").strip()
            claims_list = [c.strip("- ").strip() for c in parts[1].split("\n") if c.strip()]
        else:
            summary_text = analysis_result
            
        st.markdown(f"<div class='card'><h3>📺 영상 요약</h3>{summary_text}</div>", unsafe_allow_html=True)
        
        st.markdown("### 🕵️ 팩트체크 리포트")
        for claim in claims_list:
            if len(claim) < 5: continue
            
            try:
                search_res = search.invoke(claim)
                evidence = str(search_res)
            except:
                evidence = "검색 결과 없음"
            
            verify_prompt = PromptTemplate.from_template(
                "주장: {claim}\n증거: {evidence}\n증거를 바탕으로 이 주장이 '사실', '거짓', '판단보류' 중 무엇인지 판단하고 이유를 1문장으로 써줘."
            )
            verdict = llm.invoke(verify_prompt.format(claim=claim, evidence=evidence)).content
            
            color_class = "fact-true" if "사실" in verdict else "fact-check"
            st.markdown(f"<div class='fact-box {color_class}'><strong>🗣️ {claim}</strong><br>↳ {verdict}</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🧠 분석 로직 2: 웹 뉴스
# ---------------------------------------------------------
def analyze_article(url, llm, search):
    try:
        with st.spinner("📰 기사 본문을 읽어오는 중입니다..."):
            loader = WebBaseLoader(url)
            docs = loader.load()
            article_content = docs[0].page_content[:10000]
            article_title = docs[0].metadata.get('title', '제목 없음')
    except Exception as e:
        st.error(f"기사를 읽어올 수 없습니다: {e}")
        return

    with st.spinner("⚖️ Veritas Lens가 기사의 이면을 파헤치고 있습니다..."):
        # 1. LLM 분석
        bias_prompt = PromptTemplate.from_template("""
        기사 제목: {title}
        기사 본문: {text}
        
        다음 3가지를 분석해줘:
        1. 자극성 점수 (0~100점, 숫자만): 예) 85
        2. 이 기사의 프레이밍(의도) 요약 (2문장 이내)
        3. 검색해야 할 키워드 1개 (단어만)
        
        형식:
        SCORE: 점수
        FRAMING: 내용
        KEYWORD: 단어
        """)
        
        try:
            bias_res = llm.invoke(bias_prompt.format(title=article_title, text=article_content)).content
        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")
            return
        
        # 파싱 로직
        score = 0
        framing = "분석 실패"
        keyword = article_title
        
        for line in bias_res.split('\n'):
            if "SCORE:" in line: 
                try:
                    score_str = line.split(":")[1].strip().replace("점", "")
                    score = int(re.findall(r'\d+', score_str)[0])
                except: score = 50
            if "FRAMING:" in line: framing = line.split(":")[1].strip()
            if "KEYWORD:" in line: keyword = line.split(":")[1].strip()
            
        # 2. 외부 맥락 검색
        search_res = search.invoke(keyword)
        missing_context = llm.invoke(f"기사 내용: {article_content}\n외부 사실: {search_res}\n기사에서 누락된 중요한 맥락 1가지만 찾아서 핵심만 설명해줘.").content

        # -----------------------------------------------------
        # 🎨 UI 렌더링 (HTML/CSS Injection)
        # -----------------------------------------------------
        
        # 점수에 따른 색상 결정
        if score >= 70:
            score_color = "#ef4444" # Red
            score_msg = "🔥 매우 자극적 (High Bias)"
        elif score >= 40:
            score_color = "#f97316" # Orange
            score_msg = "⚠️ 주의 필요 (Moderate)"
        else:
            score_color = "#10b981" # Green
            score_msg = "✅ 중립적 (Neutral)"
            
        st.markdown(f"## 📰 {article_title}")
        
        # 레이아웃: 1:2 비율 (왼쪽: 스코어/요약, 오른쪽: 상세 분석)
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # [카드 1] 자극성 지수 (게이지 바 스타일)
            st.markdown(f"""
                <div style="background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; text-align: center;">
                    <div style="font-size: 14px; color: #6b7280; margin-bottom: 5px;">자극성 지수 (Bias Score)</div>
                    <div style="font-size: 48px; font-weight: 800; color: {score_color};">{score}</div>
                    <div style="background-color: #e5e7eb; border-radius: 9999px; height: 10px; width: 100%; margin: 10px 0; overflow: hidden;">
                        <div style="background-color: {score_color}; height: 100%; width: {score}%;"></div>
                    </div>
                    <div style="font-size: 14px; font-weight: bold; color: {score_color};">{score_msg}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # [카드 2] 검색 키워드
            st.markdown(f"""
                <div style="background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="font-size: 14px; color: #6b7280; margin-bottom: 10px;">검증 키워드</div>
                    <span style="background-color: #dbeafe; color: #1e40af; padding: 5px 12px; border-radius: 20px; font-weight: 600; font-size: 14px;">🔍 {keyword}</span>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            # [카드 3] 프레이밍 분석
            st.markdown(f"""
                <div style="background-color: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;">
                    <h3 style="margin-top: 0; font-size: 18px; color: #111827;">🧐 이 기사의 프레이밍(의도)</h3>
                    <p style="font-size: 16px; line-height: 1.6; color: #374151; margin-bottom: 0;">{framing}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # [카드 4] 놓친 맥락 (강조 스타일)
            st.markdown(f"""
                <div style="background-color: #fff7ed; border-left: 5px solid #f97316; padding: 25px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <h3 style="margin-top: 0; font-size: 18px; color: #9a3412;">🧩 놓친 맥락 (Missing Context)</h3>
                    <p style="font-size: 15px; line-height: 1.6; color: #7c2d12; margin-bottom: 0;">{missing_context}</p>
                    <div style="margin-top: 15px; font-size: 12px; color: #9a3412; opacity: 0.8;">
                        * AI가 외부 검색 결과를 바탕으로 보완한 정보입니다.
                    </div>
                </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 🚀 메인 실행부
# ---------------------------------------------------------
st.markdown('<div class="main-title">Veritas Lens <span style="font-size:1.5rem; color:#3B82F6;">Beta</span></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">See the truth behind the noise. URL 하나로 팩트와 편향성을 꿰뚫어 보세요.</div>', unsafe_allow_html=True)

url_input = st.text_input("🔗 분석하고 싶은 링크를 입력하세요 (YouTube or News URL)", placeholder="https://...")

if st.button("Analyze Link 🚀"):
    if not url_input:
        st.warning("링크를 입력해주세요!")
    elif not openai_api_key or not tavily_api_key:
        st.error("기본 API Key(OpenAI, Tavily) 설정이 필요합니다.")
    else:
        llm_instance = get_llm(openai_api_key)
        search_tool = get_search_tool(tavily_api_key)
        
        if "youtube.com" in url_input or "youtu.be" in url_input or "shorts" in url_input:
            # RapidAPI 키 확인
            if "RAPIDAPI_KEY" in st.secrets:
                rapid_key = st.secrets["RAPIDAPI_KEY"]
            else:
                rapid_key = st.text_input("RapidAPI Key", type="password") # Secrets에 없으면 입력받음
                
            if rapid_key:
                analyze_youtube(url_input, llm_instance, search_tool, rapid_key)
            else:
                st.error("YouTube 분석을 위해 RapidAPI Key가 필요합니다.")
        else:
            analyze_article(url_input, llm_instance, search_tool)


