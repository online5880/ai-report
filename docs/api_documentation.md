
# API 문서

## 1. 학습 리포트 생성 API

  <img src="https://github.com/online5880/ai-report/blob/main/images/api_report.png?raw=true" width="50%" height="100%"/>

- **엔드포인트**: `POST /api/streaming-daily-report/`
- **설명**: 사용자의 학습 기록을 기반으로 AI 리포트를 생성합니다.

#### 요청
- **Content-Type**: `application/json`
- **Body**:
  {
    "user_id": "string",
    "date": "2024-01-09"
  }

#### 응답
- **HTTP 상태 코드**: `200 OK`
- **Content-Type**: `application/json`
- **Body 예시**:
  {
    "status": "success",
    "report_id": "12345",
    "message": "AI 리포트가 성공적으로 생성되었습니다."
  }

#### 에러 응답
- **HTTP 상태 코드**: `400 Bad Request`
- **Body 예시**:
  {
    "status": "error",
    "message": "필수 필드가 누락되었습니다."
  }

#### 추가 설명
- **사용 사례**:
  - 특정 날짜의 학습 데이터를 바탕으로 AI 리포트를 생성하고 이를 분석에 활용.
- **주의 사항**:
  - `user_id`는 시스템에 등록된 사용자여야 합니다.
  - `date` 필드는 ISO 형식(`YYYY-MM-DD`)으로 전달해야 합니다.

---

## 2. GKT

### 1. `/health` (GET)
- **설명**: API 서버가 정상적으로 작동하는지 확인하는 엔드포인트입니다.
- **요청 데이터**: 없음
- **응답 데이터**:
    ```bash
    Status Code: 200 OK
    {
        "status": "ok"
    }
    ```

- **에러 응답**:
    ```bash
    Status Code: 500 Internal Server Error
    {
        "detail": "Error message here"
    }
    ```

---

### 2. `/api/gkt` (POST)

  <img src="https://github.com/online5880/ai-report/blob/main/images/api_gkt.png?raw=true" width="50%" height="100%"/>

- **설명**: 사용자의 문제 풀이 이력을 기반으로 다음 문제의 예측 값을 반환하는 엔드포인트입니다.
- **요청 데이터**:
    ```bash
    Content-Type: application/json
    Body:
    {
        "user_id": "string",
        "skill_list": [integer],
        "correct_list": [integer]
    }
    ```

    ```bash
    Example:
    {
        "user_id": "f70467a4-7bc5-4bda-99d4-0280065daa65",
        "skill_list": [14201779, 14201792, 14201789, 14201858, 14201862, 14201865, 14201890, 14201897, 14201905, 14201880],
        "correct_list": [1, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    }
    ```

- **응답 데이터**:
  ```bash
  Status Code: 200 OK
  {
    "predictions": [
      { "skill_code": probability }
    ]
  }
    ```
    ```bash
  Example:
  {
    "predictions": [
      { "14201779": 0.5465942025184631 },
      { "14201792": 0.6112098693847656 },
      { "14201789": 0.5456284880638123 },
      ...
    ]
  }
    ```

- **에러 응답**:
    ```bash
  Status Code: 400 Bad Request
  {
    "detail": "The skill_list and correct_list must contain at least 10 elements."
  }
  ```
  ```bash
  Status Code: 404 Not Found
  {
    "detail": "No data found for user_id: user_123"
  }
  ```
  ```bash
  Status Code: 500 Internal Server Error
  {
    "detail": "Error message here"
  }
    ```

---

### 3. `/api/gkt/confusion-matrix` (POST)

<img src="https://github.com/online5880/ai-report/blob/main/images/api_gkt_confusion.png?raw=true" width="50%" height="100%"/>

- **설명**: 사용자 데이터와 입력 데이터를 기반으로 혼동 행렬을 생성하며, 각 문제에 대한 분석 정보를 포함한 응답을 반환합니다.
- **요청 데이터**:

    ```bash
  Content-Type: application/json
  Body:
  {
    "user_id": "string",
    "skill_list": [integer],
    "correct_list": [integer]
  }
    ```
    ```bash
  Example:
  {
    "user_id": "f70467a4-7bc5-4bda-99d4-0280065daa65",
    "skill_list": [14201779, 14201792, 14201789, 14201858, 14201862, 14201865, 14201890, 14201897, 14201905, 14201880],
    "correct_list": [1, 1, 0, 0, 1, 0, 1, 0, 0, 1]
  }
  ```

- **응답 데이터**:

    ```bash
  Status Code: 200 OK
  {
    "confusion_matrix": [
      {
        "skill": integer,
        "predicted_probability": float,
        "predicted_result": integer,
        "actual_result": integer,
        "analysis": "string"
      }
    ]
  }
    ```

    ```bash
  Example:
  {
    "confusion_matrix": [
      { "skill": 14201779, "predicted_probability": 0.5465942025184631, "predicted_result": 1, "actual_result": 1, "analysis": "개념 확립 (정답 확신)" },
      { "skill": 14201792, "predicted_probability": 0.6112098693847656, "predicted_result": 1, "actual_result": 1, "analysis": "개념 확립 (정답 확신)" },
      ...
    ]
  }
  ```

- **에러 응답**:

    ```bash
    Status Code: 400 Bad Request
    {
        "detail": "One or more skills in skill_list are not present in the data."
    }
    ```

    ```bash
    Status Code: 500 Internal Server Error
    {
        "detail": "Error message here"
    }
    ```

---

### 4. `api/graphsage/recommend` (POST)

  <img src="https://github.com/online5880/ai-report/blob/main/images/api_graphsage.png?raw=true" width="50%" height="100%"/>


- **설명**: predictions에서 이해도가 낮은 개념에 대해 유사한 개념을 추천합니다. `top_k`는 최대 추천 개수를 의미합니다.
- **요청 데이터**:
    ```bash
    Content-Type: application/json
    Body:
    {
        "predictions": [ { "concept_id": float } ],
        "top_k": integer
    }
    ```
    ```bash
    Example:
    {
        "predictions": [ { "14201779": 0.546 } ],
        "top_k": 3
    }
    ```
    - **응답 데이터**:
    ```bash
    Status Code: 200 OK
    {
        "recommendations": [
        { "target": [...], "similar": [...] }
        ]
    }
    ```

- **에러 응답**:
    ```bash
    Status Code: 400 Bad Request
    {
        "detail": "The 'top_k' value must not exceed 5."
    }
    ```
    ```bash
    Status Code: 404 Not Found
    {
        "detail": "File not found: Embedding file not found at /path/to/file"
    }
    ```
    ```bash
    Status Code: 500 Internal Server Error
    {
        "detail": "Unexpected error occurred: division by zero"
    }
    ```