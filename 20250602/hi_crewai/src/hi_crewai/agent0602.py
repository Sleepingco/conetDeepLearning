# %%
from dotenv import load_dotenv
import os
print('API Key loaded',os.getenv("OPENAI_API_KEY"))

# %%
from crewai import Agent, Task, Crew, Process # CrewAI 핵심 클래스들
from dotenv import load_dotenv
import os

# 0 환경변수 로드(.env에서 OPENAI_API_KEY 불러오기)
load_dotenv() # .env 파일에 정의된 환경변수를 로드합니다.
# OPENAI_API_KEY가 환경변수로 설정되엇다면,openai패키지가 이를 자동으로 사용합니다.

# 1.Agent 생성:role,goal,backstory 설정
agent = Agent(
    role='AI 어시스턴트', # 에이전트의 역할
    goal='사용자에게 간단한 환영 인사를 제공', # 에이전트가 달성할 목표
    backstory='당신은 친절한 AI비서로, 언제나 정중하고 도움이 되는 인사를 건넵니다' # 에이전트의 성격/배경

)

# %%
# Task 생성: description, expected_outpuy,agent
task = Task(
    description='사용자에게 환영 인사 한 마디를 작성하세요', # 에이전트가 수행할 작업 내용
    expected_output='한 줄의 따뜻한 환영 메세지', # 기대하는 출력 결과물
    agent = agent
)

# %%
# 3. Crew 생성: 에이전트와 태스크를 크루로 구성 (순차 프로세스 설정)
crew = Crew(
    agents=[agent], # 크루에 속한 에이전트 목록 (여기서는 1개)
    tasks=[task], # 크루가 수행할 태스크 목록 (여기서는 1개)
    process=Process.sequential, # 순차적 진행 방식으로 태스크 실행
    verbose=False # 실행 중 상세한 로그 출력 여부
)

# %%
# 4. Crew 실행: kickoff() 메서드로 태스크 수행 시작
result = crew.kickoff()
# 5. 실행 결과 출력
print("최종 결과:", result)

# %%
# 필요한 라이브러리 임포트
from crewai import Agent, Task, Crew, Process # CrewAI 핵심 클래스들
from dotenv import load_dotenv
import os
# 0. 환경 변수 로드 (.env에서 OPENAI_API_KEY 불러오기)
load_dotenv() # .env 파일에 정의된 환경변수를 로드합니다.
# OPENAI_API_KEY가 환경변수로 설정되었다면, openai 패키지가 이를 자동으로 사용합니다.
# 1. Agent 생성: role, goal, backstory 설정
agent = Agent(
    role="{topic} AI 어시스턴트", # 에이전트의 역할
    goal="{topic} 사용자에게 간단한 환영 인사를 제공", # 에이전트가 달성할 목표
    backstory="당신은 친절한 AI 비서로, {topic}에서 언제나 정중하고 도움이 되는 인사를 건넵니다." # 에이전트의 성격/배경
)
# 2. Task 생성: description, expected_output, agent 지정
task = Task(
    description="{topic}에 참석한 사용자에게 환영 인사 한 마디를 작성하세요.", # 에이전트가 수행할 작업 내용
    expected_output="{topic}에서의 한 줄의 따뜻한 환영 메시지", # 기대하는 출력 결과물
    agent=agent # 이 태스크를 수행할 에이전트
)

# %%
# 3. Crew 생성: 에이전트와 태스크를 크루로 구성 (순차 프로세스 설정)
crew = Crew(
    agents=[agent], # 크루에 속한 에이전트 목록 (여기서는 1개)
    tasks=[task], # 크루가 수행할 태스크 목록 (여기서는 1개)
    process=Process.sequential, # 순차적 진행 방식으로 태스크 실행
    verbose=True # 실행 중 상세한 로그 출력 여부
)


# %%
# 4. Crew 실행: kickoff() 메서드로 태스크 수행 시작
inputs = {
    "topic": "결혼식"
}


# %%
result = crew.kickoff(inputs=inputs)

# %%
# 5. 실행 결과 출력
print("최종 결과:", result)

# %%
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
# 1. .env 파일에서 OpenAI API 키 불러오기
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key is None:
 raise EnvironmentError("OPENAI_API_KEY is not set in .env file. Please add your OpenAI API key to .env.")

# %%
# 2. 에이전트 정의 (연구원, 작가)
researcher = Agent(
 role="AI Researcher",
 goal="멀티 에이전트 시스템의 주요 장점을 3가지 조사하여 설명합니다.",
 backstory="다년간 AI 트렌드를 연구해온 전문가입니다."
)
writer = Agent(
 role="Technical Writer",
 goal="연구 결과를 바탕으로 간략한 결론을 작성합니다.",
 backstory="복잡한 정보를 쉽게 요약하는 작문 전문가입니다."
)

# %%
# 3. 태스크 정의 (조사 태스크, 작성 태스크)
research_task = Task(
    description="멀티 에이전트 시스템의 주요 장점 3가지를 조사하고 각 장점을 간략히 설명하세요.",
    expected_output="3가지 장점에 대한 간단한 설명 (bullet point 목록)",
    agent=researcher
)
write_task = Task(
    description="위 조사 결과를 참고하여, 멀티 에이전트 시스템의 장점에 대한 짧은 결론을 작성하세요.",
    expected_output="3~4문장으로 구성된 결론 단락",
    agent=writer,
    context=[research_task] # 이전 태스크의 결과를 활용
)

# %%
# 4. 크루 생성 (순차적 프로세스 설정)
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True
)

# %%
# 5. 크루 실행 및 결과 출력
crew_output = crew.kickoff()
# 6. 각 태스크의 결과를 순차적으로 출력
for idx, task_output in enumerate(crew_output.tasks_output, start=1):
 print(f"\n[Task {idx} Output]\n{task_output}\n")

# %%
# 단일 에이전트 (다양한 치환잔)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from crewai import  Agent,task,Crew,Process

# .env 파일 업로드하여 환경변수 설정
load_dotenv()

# %%
# 여행 기획사 에이전트 정의
travel_agent = Agent(
    role='여행 기획자',
    goal='사용자의 요청에 따라 {place} 여행 일정을 계획하고 제안합니다.',
    backstory='여행사에서 10년 경력의 전문 여행 플래너로, 다양한 국내 여행 코스를 알고 있습니다.',
    verbose=False
)

# %%
# 부산 3일 여행 일정 작성 Task 정의
itinerary_task = Task(
    description=(
        "{place}에서 {days}일간 여행 일정을 계획해 주세요.\n"
        "{days}일을 1일차, 2일차와 같이 차수로 나누고, 각 일차 마다 아침/점심/저녁에 할 활동을 상세히 제안하세요.\n"
        "여행 일정에는 {place}의 주요 관광지와 현지 맛집 추천을 포함하고, 교통 수단 정보나 팁이 있으면 함께 제공하세요."
    ),
    agent=travel_agent,
    expected_output='{days}일을 Day 1, Day 2,...으로 구분된 상세 일정 제안'
)

# %%
# Crew 생성 및 실행(순차 실행-Task가 하나뿐이므로 순차 처리)
crew_single = Crew(
    agents=[travel_agent],
    tasks=[itinerary_task],
    process=Process.sequential,
    verbose=True
)


# %%
print('===[단일 에이전트] 부산 3일 일정 생성 시작===')
result_single = crew_single.kickoff(inputs={"place":"부산","days":3})
print("=== [단일 에이전트] 생성된 부산 3일 일정 ===")
print(result_single)

# %%
from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from crewai import Agent, Task, Crew, Process

load_dotenv()

# %%
# 커스텀 웹 검색 도구 준비
class MyCustomDuckDuckGoTool(BaseTool):
    name:str='DuckDuckGo Search Tool'
    description:str = "웹에서 최신 정보를 검색할 수 있는 도구입니다."

    def _run(self,query:str) -> str:
        duckduckgo_tool = DuckDuckGoSearchRun()
        response = duckduckgo_tool.invoke(query)
        return response

# DuckDuckGo 검색 도구 인스턴스 생성
search_tool =  MyCustomDuckDuckGoTool()

# %%
# 정보 조사 에이전트 정의
research_agent=Agent(
    role='정보 조사자',
    goal='{place} 여행에 필요한 최신 정보를 조사하여 제공합니다',
    backstory='온라인 정보 검색에 능총한 여행 정보 전문가입니다.',
    tools=[search_tool], # 웹 검색 도구 장착
    verbose=True
)

# 일정 작성 에이전트 정의
planner_agent=Agent(
    role='여행 일정 기획자',
    goal='제공된 정보를 활용해 완성도 높은 {place} 여행 일정을 작성합니다.',
    backstory='국내 여행 일정을 여러 차례 기획한 경험이 풍부한 전문가입니다.',
    verbose=True
)

# %%
# 정보 조사 Task 정의
research_task = Task(
    description=(
        '{place} 여행을 위해 알아야 할 핵심 정보를 조사하세요.\n'
        '{place}의 인기 관광지 목록, 지역별 맛집 추천, 이동 시 유용한 교통 정보 등을 최신 자료를 기반으로 정리해 주세요'
    ),
    agent=research_agent,
    expected_output='한국어로 작성된 {place} 여행에 대한 요약 정보 목록'
)

# 일정 작성 Task 정의(이전 Task 결과를 context로 활용)
planning_task = Task(
    description=(
         "위의 조사 결과를 참고하여 {place}에서 {days}일 동안 머무는 여행 일정을 작성해 주세요.\n"
        "각 날짜별로 오전/오후/저녁 계획을 세우고, 조사된 관광지와 맛집 정보를 일정에 반영하세요.\n"
        "일정에는 방문지에 대한 간단한 설명이나 여행 팁도 포함해 주세요."
    ),
    agent=planner_agent,
    context=[research_task], # 이전 조사 결과를 컨텍스트로 전달
    expected_output='한국어로 작성된 조사된 정보를 반영한{days}일간의 여행 일정'
)


# %%
# 두 에이전트를 Crew로 묶어 순차 실행
crew_multi = Crew(
    agents=[research_agent,planner_agent],
    tasks=[research_task,planning_task],
    process=Process.sequential,
    verbose=True
)
print("\n=== [협업 에이전트] 부산 3일 일정 생성 시작 ===")
result_multi = crew_multi.kickoff(inputs={"place": "부산", "days": 3})
print("=== [협업 에이전트] 생성된 부산 3일 일정 ===")
print(result_multi)

# %%
# 개선된 일정 작성 Task 정의(예산 및 교통 고려 추가)
imporved_planning_task = Task(
    description=(
        "위의 조사 결과를 참고하여 부산에서 3일 동안 머무는 여행 일정을 작성해 주세요.\n"
        "각 날짜별로 오전/오후/저녁 계획을 세우고, 조사된 관광지와 맛집 정보를 일정에 반영하세요.\n"
        "가능하면 예산은 하루 10만원 내외로 맞추고, 이동은 모두 대중교통 을 이용하는 것으로 고려하세요.\n"
        "버스정류장 및 지하철역을 포함한 대중교통 경로를 제안해 주세요.\n"
        "버스 및 지하철을 이용할 때의 소요 시간도 포함해 주세요.\n"
        "일정에는 방문지에 대한 간단한 설명이나 여행 팁도 포함해 주세요. 결과는 한국어로 작성해 주세요.\n"
    ),
    agent=planner_agent,
    context=[research_task],
    expected_output="예산과 교통을 고려한 3일간의 여행 일정"
)

# %%
from agent_tools import DuckDuckGoSearchTool, WebScraperTool, CalculatorTool
from crewai import Agent, Task, Crew, Process
# 도구 인스턴스 생성
search_tool = DuckDuckGoSearchTool()
scrape_tool = WebScraperTool()
calculator_tool = CalculatorTool()
news_agent = Agent(
    role="뉴스 분석가",
    goal="최신 뉴스를 검색하여 요약",
    backstory="뉴스 분석 전문 AI 기자",
    tools=[search_tool, scrape_tool],
    verbose=False
)
news_task = Task(
    description="최근 인공지능 관련 뉴스 3건을 요약하고 링크 제공",
    expected_output="한국어로 작성된 뉴스 3건의 요약 및 링크",
    agent=news_agent
)
crew = Crew(
    agents=[news_agent],
    tasks=[news_task],
    process=Process.sequential,
    verbose=False
)
result = crew.kickoff()
print(result)

# %%
from agent_tools import DuckDuckGoSearchTool, WebScraperTool, CalculatorTool
from crewai import Agent, Task, Crew, Process
# 도구 인스턴스 생성
search_tool = DuckDuckGoSearchTool()
scrape_tool = WebScraperTool()
calculator_tool = CalculatorTool()
recipe_agent = Agent(
role="요리 분석가",
    goal="요리 레시피 추천 및 요약",
    backstory="글로벌 요리 전문 AI 셰프",
    tools=[search_tool, scrape_tool],
    verbose=False
)
recipe_task = Task(
    description="채식 파스타 레시피 추천, 재료 및 조리법 안내",
    expected_output="한국어로 작성된 채식 파스타 레시피(재료 및 조리법)",
    agent=recipe_agent
)
crew = Crew(
    agents=[recipe_agent],
    tasks=[recipe_task],
    process=Process.sequential,
    verbose=False
)
result = crew.kickoff()
print(result)

# %%
from agent_tools import DuckDuckGoSearchTool, WebScraperTool, CalculatorTool
from crewai import Agent, Task, Crew, Process
# 도구 인스턴스 생성
search_tool = DuckDuckGoSearchTool()
scrape_tool = WebScraperTool()
calculator_tool = CalculatorTool()
travel_agent = Agent(
role="여행 전문가",
    goal="최적의 여행 일정과 예산 계획 제공",
    backstory="다년간의 여행 플래너 경험 보유",
    tools=[search_tool, scrape_tool, calculator_tool],
    verbose=False
)
travel_task = Task(
    description="파리 5일 여행 일정(문화, 미식 포함), 예산은 1000달러 (항공 300달러, 숙박 하루 100달러).",
    expected_output="한국어로 작성된 5일간 파리 여행에 대한 상세한 일정과 예산 계산 결과",
    agent=travel_agent
)
crew = Crew(
    agents=[travel_agent],
    tasks=[travel_task],
    process=Process.sequential,
    verbose=False
)
result = crew.kickoff()
print(result)

# %%
from crewai import Agent,Task,Crew,Process
from agent_tools import DuckDuckGoSearchTool,WebScraperTool,CalculatorTool

search_tool=DuckDuckGoSearchTool()
scrape_tool=WebScraperTool()
calculator_tool=CalculatorTool()

planner_agent = Agent(
    role="총괄 여행 플래너",
    goal="최적의 서울 근교 1박 2일 여행 일정과 추천 음식을 종합하여 최종 여행 계획서를 작성",
    backstory="10년 경력의 베테랑 여행 컨설턴트로 다양한 분야의 의견을 취합하여 여행 계획을 완성합니다.",
    allow_delegation=True, # 다른 에이전트에게 업무 위임을 허용
    verbose=False
)

# %%
# 여행 전문가 Agent (여행 일정 추천 담당)
travel_agent = Agent(
    role="여행 전문가",
    goal="최신 여행 트렌드를 조사하여 서울 근교의 인기 있는 여행지와 일정을 제안",
    backstory="국내 여행지에 대해 잘 알고 있는 전문가로, 최근의 여행 트렌드를 바탕으로 관광지를 추천합니다.",
    tools=[search_tool],
    verbose= False
)
# 요리 전문가 Agent (음식 추천 담당)
culinary_agent = Agent(
    role="요리 전문가",
    goal="추천된 여행지와 잘 어울리는 현지 음식 및 레시피를 추천",
    backstory="국내 각 지역의 음식 문화와 레시피에 능통한 전문가로, 여행지에 어울리는 음식을 추천합니다.",
    tools=[search_tool],
    verbose= False
)

# %%
# Task 정의 (총괄 플래너에게 최종 여행 계획서 작성 지시)
planner_task = Task(
    description=(
        "국내 최신 여행 트렌드가 반영된 서울 근교의 1박 2일 여행 일정을 작성하고, "
        "각 여행지와 잘 어울리는 현지 음식과 레시피를 포함하여 여행 계획서를 한국어로 작성해주세요."
    ),
    expected_output=(
        "최신 여행 트렌드를 반영한 서울 근교 1박 2일 여행 일정과 "
        "각 여행지의 현지 음식 및 간단한 레시피를 포함한 한국어 여행 계획서"
    ),
    agent=planner_agent
)

# %%
# Crew 구성 (계층적 프로세스 사용)
crew = Crew(
    agents=[travel_agent, culinary_agent], # 하위 실행에 참여할 에이전트들 (관리자 제외)
    tasks=[planner_task],
    process=Process.hierarchical,
    manager_agent=planner_agent, # 총괄 여행 플래너를 매니저로 지정
)
# Crew 실행
if __name__ == "__main__":
    result = crew.kickoff()
    print("\n\n 최종 여행 계획서:\n")
    print(result)


