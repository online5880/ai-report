# KST 이론 기반 AI 리포트

**지식공간 이론(Knowledge Space Theory, KST)**을 기반으로 학습자 개개인의 학습 결손을 추적하고 맞춤형 학습 경로를 제공합니다.

---

## 목차
- [KST 이론 기반 AI 리포트](#kst-이론-기반-ai-리포트)
  - [목차](#목차)
  - [프로젝트 소개](#프로젝트-소개)
    - [프로젝트 기간](#프로젝트-기간)
    - [프로젝트 목적](#프로젝트-목적)
    - [주요 기능](#주요-기능)
  - [기술 스택](#기술-스택)
  - [R\&R](#rr)
  - [CI/CD](#cicd)
  - [아키텍처](#아키텍처)
    - [아키텍처 다이어그램](#아키텍처-다이어그램)
  - [API 문서](#api-문서)
  - [결과 이미지 및 예시 영상](#결과-이미지-및-예시-영상)
  - [](#)
  - [데이터 보호 관련 안내](#데이터-보호-관련-안내)

---

## 프로젝트 소개

### 프로젝트 기간
- **시작일**: 2024-12-02
- **종료일**: 2025-01-10
- **총 기간**: 1개월

### 프로젝트 목적
2022년 개정 초등학교 1, 2학년 수학 교육과정을 기반으로 학습 데이터를 분석하여 다음을 목표로 합니다:
- 학습 결손 추적 및 분석
- 학습자 맞춤 학습 경로 추천
- 학습 성취도 향상

### 주요 기능
- 학습 결손 분석 및 추적
- 학습자 맞춤 학습 경로 추천
- 초등 수학 지식맵 시각화
- LLM 기반 실시간 학습 리포트 생성

---

## 기술 스택

<table>
    <tr>
        <th align="center">언어</th>
        <td>
            <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" alt="Python Badge">
            <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black" alt="JavaScript Badge">
            <img src="https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white" alt="HTML Badge">
            <img src="https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white" alt="CSS Badge">
        </td>
    </tr>
    <tr>
        <th align="center">프레임워크</th>
        <td>
            <img src="https://img.shields.io/badge/Django-092E20?style=flat&logo=django&logoColor=white" alt="Django Badge">
            <img src="https://img.shields.io/badge/DRF-ff1709?style=flat&logo=django&logoColor=white" alt="Django Rest Framework Badge">
            <img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI Badge">
            <img src="https://img.shields.io/badge/LangChain-1D3557?style=flat&logo=python&logoColor=white" alt="LangChain Badge">
        </td>
    </tr>
    <tr>
        <th align="center">데이터베이스</th>
        <td>
            <img src="https://img.shields.io/badge/PostgreSQL-336791?style=flat&logo=postgresql&logoColor=white" alt="PostgreSQL Badge">
            <img src="https://img.shields.io/badge/Neo4j-008CC1?style=flat&logo=neo4j&logoColor=white" alt="Neo4j Badge">
        </td>
    </tr>
    <tr>
        <th align="center">머신러닝 및 데이터 처리</th>
        <td>
            <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch Badge">
            <img src="https://img.shields.io/badge/Polars-FF6F00?style=flat&logo=polars&logoColor=white" alt="Polars Badge">
            <img src="https://img.shields.io/badge/OpenAI-GPT-412991?style=flat&logo=openai&logoColor=white" alt="OpenAI GPT Badge">
        </td>
    </tr>
    <tr>
        <th align="center">인프라</th>
        <td>
            <img src="https://img.shields.io/badge/AWS-232F3E?style=flat&logo=amazon-aws&logoColor=white" alt="AWS Badge">
            <img src="https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white" alt="Docker Badge">
            <img src="https://img.shields.io/badge/Nginx-009639?style=flat&logo=nginx&logoColor=white" alt="Nginx Badge">
        </td>
    </tr>
    <tr>
        <th align="center">CI/CD</th>
        <td>
            <img src="https://img.shields.io/badge/GitHub%20Actions-2088FF?style=flat&logo=github-actions&logoColor=white" alt="GitHub Actions Badge">
            <img src="https://img.shields.io/badge/AWS%20ECR-FF9900?style=flat&logo=amazon-ecr&logoColor=white" alt="AWS ECR Badge">
            <img src="https://img.shields.io/badge/AWS%20ECS-FF9900?style=flat&logo=amazon-ecs&logoColor=white" alt="AWS ECS Badge">
        </td>
    </tr>
    <tr>
        <th align="center">도구</th>
        <td>
            <img src="https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white" alt="Git Badge">
            <img src="https://img.shields.io/badge/Postman-FF6C37?style=flat&logo=postman&logoColor=white" alt="Postman Badge">
            <img src="https://img.shields.io/badge/Swagger-85EA2D?style=flat&logo=swagger&logoColor=white" alt="Swagger Badge">
            <img src="https://img.shields.io/badge/Jira-0052CC?style=flat&logo=jira&logoColor=white" alt="Jira Badge">
            <img src="https://img.shields.io/badge/Confluence-172B4D?style=flat&logo=confluence&logoColor=white" alt="Confluence Badge">
            <img src="https://img.shields.io/badge/Slack-4A154B?style=flat&logo=slack&logoColor=white" alt="Slack Badge">
        </td>
    </tr>
</table>

---

## R&R

| **역할**   | **이름**    | **책임 및 역할**                                                                                 |
|------------|-------------|-------------------------------------------------------------------------------------------------|
| **팀장**   | 박만혜      | - 시스템 아키텍처 및 AWS 클라우드 설계 (Nginx 포함) <br> - 무중단 CI/CD 파이프라인 구축 (GitHub Actions) <br> - Docker 기반 서비스(Nginx, Neo4j 등) 구축 <br> - Django REST API 설계 및 개발 <br> - 학습 리포트 개발 (LangChain, OpenAI) <br> - 그래프 데이터 처리 (GraphRAG) <br> - AWS RDS, EC2 구축 및 Nginx 로드 밸런싱 <br> - 협업 환경 관리 (Slack, Jira) |
| **팀원**   | 곽태경      | - 지식 추적 모델 설계 및 학습 <br> - MLflow 기반 모델 관리 <br> - 데이터 전처리 및 API 설계          |
| **팀원**   | 이수지      | - 학습 리포트 화면 설계 및 개발 <br> - API 설계 <br> - 학습 성과 시각화 구현                       |
| **팀원**   | 조아라      | - 추천 모델 설계 및 학습 <br> - 추천 모델 API 작성 및 성능 평가 <br> - 데이터 전처리                  |

---

## CI/CD
DevOps를 활용한 효율적인 CI/CD 파이프라인을 구축했으며, **무중단 배포(Rolling Deployment)** 방식으로 안정성을 유지합니다.

더 자세한 내용은 [CI/CD 세부 문서](docs/cicd.md)를 참조하세요.

<img src="https://github.com/online5880/ai-report/blob/main/images/cicd.png?raw=true" width="100%" height="100%"/>

---

## 아키텍처

### 아키텍처 다이어그램
| **상세 설계** | **간소화 설계** |
|---------------|----------------|
| <img src="https://github.com/online5880/ai-report/blob/main/images/architecture_01.png?raw=true" width="100%" height="100%"/> | <img src="https://github.com/online5880/ai-report/blob/main/images/architecture_02.png?raw=true" width="100%" height="100%"/> |

전체 설명은 [아키텍처 세부 문서](docs/architecture.md)를 참조하세요.

---

## API 문서
주요 API에 대한 문서는 별첨으로 제공됩니다.
- [API 문서 보기](docs/api_documentation.md)

---

## 모델 저장소
- [GKT](https://github.com/oxorudo/GKT_model)
- [GraphSAGE](https://github.com/online5880/ai-report/tree/main/graphsage_fixed_edge)

---

## 결과 이미지 및 예시 영상

- **결과 이미지**
- 캘린더

  <img src="https://github.com/online5880/ai-report/blob/main/images/calendar.png?raw=true" width="75%" height="100%"/>

- 지식맵

  <img src="https://github.com/online5880/ai-report/blob/main/images/km.png?raw=true" width="75%" height="100%"/>

- 형성평가

  <img src="https://github.com/online5880/ai-report/blob/main/images/test.png?raw=true" width="75%" height="100%"/>

- 리포트

  <img src="https://github.com/online5880/ai-report/blob/main/images/report.png?raw=true" width="75%" height="100%"/>

---

- **예시 영상**

  [![예시 영상 보기](https://img.youtube.com/vi/OMeEZKgsayc/0.jpg)](https://youtu.be/OMeEZKgsayc)
  
---

## 데이터 보호 관련 안내

> 본 프로젝트는 학습자 데이터를 사용하여 구현되었으며, 데이터의 민감성으로 인해 **중요 데이터는 외부로 유출되지 않으며, 재현이 불가능**합니다.
