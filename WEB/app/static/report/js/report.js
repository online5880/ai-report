document.addEventListener("DOMContentLoaded", function () {
    const calendarEl = document.getElementById("calendar");
    const calendar = new FullCalendar.Calendar(calendarEl, {
        initialView: "dayGridMonth",
        initialDate: "2024-01-01",  // 원하는 연도와 월을 설정 (예: 2025년 1월)
        locale: "ko",
        headerToolbar: {
            left: "prev,next today",
            center: "title",
            right: "dayGridMonth,dayGridWeek,dayGridDay",
            eventDisplay: 'block',
            height: 'auto',
        },
        events: calendarEvents,

        eventClick: function (info) {
            fetchLLMReport(info.event.startStr);
            const clickedDate = info.event.startStr; // 클릭한 날짜 (ISO 형식)
            loadAccuracyChart(clickedDate); // 클릭한 날짜를 기준으로 그래프 로드
        },
        eventDidMount: function(info) {
            console.log('Event props:', info.event.extendedProps);

            if (info.event.extendedProps.hasReport === true) {
                info.el.classList.add('has-report');
                info.el.classList.remove('no-report');
            } else {
                info.el.classList.add('no-report');
                info.el.classList.remove('has-report');
            }
        },
    });
    calendar.render();

    // goToMonth 함수를 window 객체에 등록
    window.goToMonth = function (month) {
        const date = new Date();
        date.setMonth(month);
        calendar.gotoDate(date);
    };

    // 모달 관련 코드
    const modal = document.getElementById("reportModal");
    const knowledgeMapModal = document.getElementById("knowledgeMapModal");
    const span = document.getElementsByClassName("close");

    let abortController = null; // AbortController 인스턴스 저장

    // 모달 닫기 버튼 이벤트 리스너 추가
    Array.from(span).forEach(function(element) {
        element.onclick = function () {
            if (abortController) {
                abortController.abort(); // 요청 취소
            }
            modal.style.display = "none";
            knowledgeMapModal.style.display = "none";
        };
    });

    window.onclick = function (event) {
        if (event.target == modal || event.target == knowledgeMapModal) {
            if (abortController) {
                abortController.abort(); // 요청 취소
            }
            modal.style.display = "none";
            knowledgeMapModal.style.display = "none";
        }
    };

    // fetchLLMReport 함수도 window에 등록
    window.fetchLLMReport = function (date) {
        const output = document.getElementById("llm-output");
        output.innerHTML = "불러오는 중...";
        modal.style.display = "block";

        if (abortController) {
            abortController.abort();
        }

        abortController = new AbortController();

        async function fetchReportStream() {
            try {
                const response = await fetch(`/api/streaming-daily-report/`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        user_id: userId,  // calendar.html에서 전달받은 userId 사용
                        date: date
                    }),
                    signal: abortController.signal
                });

                if (!response.ok) {
                    throw new Error(`서버 오류: ${response.status} ${response.statusText}`);
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder("utf-8");

                let accumulatedText = "";
                let chunkBuffer = "";

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value, { stream: true });
                    chunkBuffer += chunk;
                    accumulatedText += chunkBuffer;

                    updateOutput(output, accumulatedText);

                    chunkBuffer = "";
                }

                // 리포트 내용 업데이트
                output.innerHTML = marked.parse(accumulatedText);

            } catch (error) {
                if (error.name !== 'AbortError') { // AbortError는 무시
                    console.error("스트리밍 오류:", error);
                    output.innerHTML = `<p style="color:red;">오류 발생: ${error.message}</p>`;
                }
            } finally {
                abortController = null; // 요청 완료 후 AbortController 초기화
            }
        }

        function updateOutput(element, content) {
            requestAnimationFrame(() => {
                element.innerHTML = marked.parse(content);
            });
        }

        fetchReportStream();

        // 지식 그래프 로딩
        fetchAndRenderKnowledgeGraph(date);
    };

    const titleElement = document.querySelector('.fc-toolbar-title');
    if (titleElement) {
        titleElement.style.cursor = 'pointer';
        flatpickr(titleElement, {
            locale: 'ko',
            dateFormat: 'Y-m',
            onChange: function(selectedDates) {
                calendar.gotoDate(selectedDates[0]);
            }
        });
    }

    // 정답률 그래프 로드 및 업데이트
    async function loadAccuracyChart(targetDate) {
        const ctx = document.getElementById('accuracy-chart').getContext('2d');

        try {
            const response = await fetch('/api/accuracy/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: userId, // calendar.html에서 전달받은 userId 사용
                    target_date: targetDate, // 클릭한 날짜
                }),
            });

            if (!response.ok) {
                throw new Error('정답률 데이터를 가져오는 데 실패했습니다.');
            }

            const data = await response.json();

            // 클릭한 날짜를 기준으로 5일간의 날짜 배열 생성
            const baseDate = new Date(targetDate);
            const dates = Array.from({ length: 5 }, (_, i) => {
                const currentDate = new Date(baseDate);
                currentDate.setDate(baseDate.getDate() + (i - 2)); // 기준 날짜를 중심으로 -2일 ~ +2일
                return `${currentDate.getMonth() + 1}/${currentDate.getDate()}`; // "MM/DD" 형식
            });

            // 기존 Chart 인스턴스가 있다면 제거
            if (window.accuracyChart) {
                window.accuracyChart.destroy();
            }

            // 서버에서 반환된 데이터로 그래프 업데이트
            window.accuracyChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: dates, // ['01/17', '01/18', '01/19', '01/20', '01/21']
                    datasets: [{
                        label: '정답률 (%)',
                        data: data.data, // [100, 100, 100, 0, 0]
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.1,
                        pointBackgroundColor: dates.map((_, index) => {
                            // 클릭한 날짜만 빨간색
                            return index === 2 ? 'rgba(255, 99, 132, 1)' : 'rgba(75, 192, 192, 1)';
                        }),
                        pointRadius: dates.map((_, index) => {
                            // 클릭한 날짜만 점 크기를 크게
                            return index === 2 ? 8 : 4;
                        }),
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    layout: {
                        padding: {
                            top: 20, // 그래프 상단 여백
                            bottom: 10,
                        },
                    },
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top',
                        },
                        title: {
                            display: true,
                            text: '정답률 변화', // 그래프 제목
                            font: {
                                size: 20,
                                weight: 'bold',
                            },
                        },
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            suggestedMax: 110, // Y축축 최대값 설정
                            ticks: {
                                stepSize: 10, // Y축 간격 설정
                                callback: function(value) {
                                    // 110인 경우 빈 문자열 반환 (라벨 숨기기)
                                    if (value === 110) return '';
                                    return value;
                                },
                            },
                        },
                    },
                },
            });
        } catch (error) {
            console.error('정답률 그래프 로딩 실패:', error);
        }
    }

        // 전체 지식 맵 버튼 클릭 이벤트 리스너 추가
        const showKnowledgeMapBtn = document.getElementById("showKnowledgeMapBtn");
        showKnowledgeMapBtn.addEventListener("click", function () {
            knowledgeMapModal.style.display = "block";
            fetchAndRenderKnowledgeGraph();
        });

        async function fetchAndRenderKnowledgeGraph() {
        try {
            // 기존 그래프 제거
            const graphContainer = document.getElementById('knowledge-graph');
            graphContainer.innerHTML = ''; // 기존 내용을 초기화

            // PyVis 임베딩 (iframe 방식)
            const iframe = document.createElement('iframe');
            iframe.src = pyvisHtmlPath; // Django에서 전달된 경로 사용
            iframe.width = '100%';
            iframe.height = '100%';
            iframe.style.border = 'none';

            // iframe 추가
            // graphContainer.appendChild(iframe);

        } catch (error) {
            console.error('지식 그래프 로딩 실패:', error);
            document.getElementById('knowledge-graph').innerHTML =
                '<p style="color: red;">지식 그래프를 불러오는데 실패했습니다.</p>';
        }
    }
});