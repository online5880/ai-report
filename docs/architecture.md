### 아키텍처 설명

AWS 클라우드 상에서 웹 애플리케이션, 데이터베이스, 머신러닝/AI 서비스를 컨테이너 형태로 통합 구조로 설계되었습니다.

주요 구성 요소는 다음과 같습니다:

---

#### **VPC(가상 사설망)**
- AWS 리전에 생성된 VPC 내부에 **Public Subnet**과 **Private Subnet**으로 나뉘어 네트워크가 구성됩니다.
  - **Public Subnet**: 인터넷과 직접 통신이 필요한 EC2 인스턴스, NGINX 등이 위치
  - **Private Subnet**: 데이터베이스(RDS)와 내부 서비스가 위치하며, 외부 접근이 제한

---

#### **EC2 인스턴스 (Public Subnet)**
- 여러 Docker 컨테이너를 통해 다음 서비스를 운영:
  - **Django**: 웹 애플리케이션
  - **FastAPI**: 그래프 알고리즘, AI inference API
  - **MLFlow Server**: 머신러닝 모델 관리
  - **Neo4j**: 그래프 데이터베이스
  - **GPT, LangChain**: AI 모델 및 LLM 기반 기능

- Docker Compose를 사용해 컨테이너 간의 효율적인 실행 및 관리가 이루어짐

---

#### **RDS (Private Subnet)**
- 관계형 데이터베이스(예: PostgreSQL)
- Private Subnet에 위치하여 외부 접근을 제한, Django 등 웹 애플리케이션과 내부 네트워크로 통신

---

#### **NGINX**
- 리버스 프록시 역할 수행:
  - 사용자 요청(80번 포트, HTTP)을 수신하여 내부 서비스로 전달
  - Gunicorn(또는 uWSGI)와 같은 애플리케이션 서버로 트래픽을 분산
  - 트래픽 분산 및 보안 관리를 통해 안정적이고 효율적인 네트워크 운영 가능

---

#### **Gunicorn + Load Balancer**
- **Gunicorn**: Python 기반 애플리케이션 서버로 Django, FastAPI 구동
- **Load Balancer**:
  - EC2 내부의 여러 컨테이너(8001, 8002, 8003 등)로 트래픽을 분산
  - NGINX로부터 전달받은 요청을 적절히 처리

---

#### **Django, FastAPI, 기타 AI 관련 컨테이너**
- **Django**: 사용자 인증, 웹 애플리케이션 기능 제공
- **GPT, LangChain**: 대형 언어 모델(LLM) 기반 AI 서비스
- **MLFlow Server**: 모델 학습 이력 관리
- **Neo4j**: 지식맵 데이터 저장소로 활용
- **FastAPI**: 모델 API 제공

---

#### **인터넷 게이트웨이와 사용자**
- 최종 사용자는 브라우저 또는 클라이언트를 통해 다음 경로로 요청을 보냅니다:
  - **인터넷 게이트웨이 → NGINX(EC2) → 로드 밸런서/Gunicorn → 각 서비스**

- 외부 서비스(Public Subnet)는 NGINX를 통해 접근 가능하며, 민감한 데이터베이스는 Private Subnet에 배치되어 보안이 강화됩니다.

---

### 요약
- **AWS VPC**: Public Subnet에는 EC2(도커 컨테이너)와 NGINX, Private Subnet에는 RDS 배치
- **EC2 컨테이너**: Django, FastAPI, MLFlow, Neo4j, GPT/LangChain 등 다양한 기능을 통합 실행
- **NGINX와 로드 밸런서**: 트래픽을 적절히 분산, 사용자 요청 처리 효율화
- **안정성**: AI 모델, 웹 서비스, DB가 유기적으로 연결되어 무중단 운영 가능