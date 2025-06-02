import streamlit as st
from datetime import datetime
from crew import HiCrewaiSequential, HiCrewaiHierarchy  # 두 가지 구조로 정의

st.set_page_config(page_title="대통령 후보 공약 분석기", layout="centered")
st.title("🗳️ 대통령 후보 공약 비교 분석기")

st.markdown("후보, 이슈, 지역, 분석 관점을 선택하고 '분석 시작' 버튼을 눌러주세요.")

# 후보 선택
candidates = st.multiselect(
    "21대 대통령 후보 선택",
    ['이재명','김문수','이준석','권영국','송진호'],
    default=[]
)

# 이슈 선택
issue = st.selectbox(
    "관심 정책 이슈",
    ["부동산", "복지", "교육", "국방", "노동", "세금", "기후 변화"]
)

# 지역 선택
region = st.selectbox(
    "거주 지역 또는 관심 지역",
    ["전국 단위", "서울/수도권", "부산/경남", "대구/경북", "호남권", "충청권", "강원/제주"]
)

# 관점 선택
perspective = st.radio(
    "정책 분석 관점",
    ["중립", "비판적 관점", "지지적 관점"]
)

# 실행 방식 선택
mode = st.radio("에이전트 실행 구조 선택", ["순차적 (Sequential)", "계층적 (Hierarchy)"])

if st.button("🔍 공약 분석 시작"):
    if not candidates:
        st.warning("최소 한 명 이상의 후보를 선택하세요.")
    else:
        st.info("에이전트가 공약 분석을 시작합니다... 잠시만 기다려주세요.")

        inputs = {
            "candidates": candidates,
            "issue": issue,
            "region": region,
            "perspective": perspective,
            "current_year": str(datetime.now().year)
        }

        try:
            if mode == "순차적 (Sequential)":
                result = HiCrewaiSequential().crew().kickoff(inputs=inputs)
            else:
                result = HiCrewaiHierarchy().crew().kickoff(inputs=inputs)

            st.success("✅ 분석이 완료되었습니다!")
            st.markdown("## 📄 분석 결과")

            # 최종 요약 결과
            if hasattr(result, "raw") and isinstance(result.raw, str):
                st.markdown("### 🧾 최종 요약")
                st.markdown(result.raw)

            # 작업별 출력
            if hasattr(result, "tasks_output") and isinstance(result.tasks_output, list):
                st.markdown("### 📊 작업별 분석 결과")

                import re
                def parse_task_string(task_str):
                    try:
                        name = re.search(r"name='(.*?)'", task_str).group(1)
                        agent = re.search(r"agent='(.*?)'", task_str).group(1)
                        raw = re.search(r"raw=(?:'|\")(.+?)(?:'|\")\s*,\s*pydantic=", task_str, re.DOTALL).group(1)
                        raw = raw.replace("\\n", "\n").replace("\\'", "'").strip()
                        return name, agent, raw
                    except Exception:
                        return None, None, None

                for idx, task in enumerate(result.tasks_output):
                    if isinstance(task, str) and "TaskOutput" in task:
                        name, agent, raw_text = parse_task_string(task)
                        if name and agent and raw_text:
                            with st.expander(f"📌 {idx+1}. {name} ({agent})", expanded=False):
                                st.markdown(raw_text)

            # 다운로드 링크
            report_path = "final_report.md"
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    report_text = f.read()
                st.download_button("📥 보고서 다운로드", report_text, file_name="대통령후보_보고서.md")
            except FileNotFoundError:
                st.warning("❗ 보고서 파일이 존재하지 않습니다.")

        except Exception as e:
            st.error(f"❌ 에러 발생: {e}")
