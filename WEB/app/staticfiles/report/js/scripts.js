document.addEventListener("DOMContentLoaded", function () {
    const calendarEl = document.getElementById("calendar");
    const calendar = new FullCalendar.Calendar(calendarEl, {
        initialView: "dayGridMonth",
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
        const currentYear = new Date().getFullYear();
        calendar.gotoDate(new Date(currentYear, month, 1));
    };

    // 모달 관련 코드
    const modal = document.getElementById("reportModal");
    const span = document.getElementsByClassName("close")[0];

    let abortController = null; // AbortController 인스턴스 저장

    span.onclick = function () {
        if (abortController) {
            abortController.abort(); // 요청 취소
        }
        modal.style.display = "none";
    };

    window.onclick = function (event) {
        if (event.target == modal) {
            if (abortController) {
                abortController.abort(); // 요청 취소
            }
            modal.style.display = "none";
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

    // Chart.js 사용 예제
    const ctx = document.getElementById('accuracy-chart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['월', '화', '수', '목', '금'],
            datasets: [{
                label: '정답률 (%)',
                data: [85, 90, 87, 92, 88], // 예시 데이터
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 2,
                fill: false,
                tension: 0.1,
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
});

// 지식 그래프 관련 코드
async function fetchAndRenderKnowledgeGraph() {
    try {
        // 기존 그래프 제거
        d3.select('#knowledge-graph svg').remove();

        // 날짜 없이 API 호출
        const response = await fetch(`/api/knowledge-graph/`);
        const graphData = await response.json();

        const width = document.getElementById('knowledge-graph').clientWidth;
        const height = 400;

        // SVG 요소 생성
        const svg = d3.select('#knowledge-graph')
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .call(d3.zoom().on("zoom", function (event) {
                svg.attr("transform", event.transform);
            }))
            .append('g'); // 그룹 요소 추가

        // 시뮬레이션 설정
        const simulation = d3.forceSimulation(graphData.nodes)
            .force('link', d3.forceLink(graphData.links).id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-200))
            .force('center', d3.forceCenter(width / 2, height / 2));

        // 링크 그리기
        const link = svg.append('g')
            .selectAll('line')
            .data(graphData.links)
            .enter()
            .append('line')
            .attr('stroke', d => d.color)
            .attr('stroke-width', 2)
            .attr('marker-end', 'url(#arrowhead)');

        // 화살표 마커 정의
        svg.append('defs').append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '-0 -5 10 10')
            .attr('refX', 15)
            .attr('refY', 0)
            .attr('orient', 'auto')
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('xoverflow', 'visible')
            .append('svg:path')
            .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
            .attr('fill', '#999')
            .style('stroke', 'none');

        // 노드 그리기
        const node = svg.append('g')
            .selectAll('circle')
            .data(graphData.nodes)
            .enter()
            .append('circle')
            .attr('r', 8)
            .attr('fill', d => d.color)
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));

        // 노드 레이블
        const label = svg.append('g')
            .selectAll('text')
            .data(graphData.nodes)
            .enter()
            .append('text')
            .text(d => d.label)
            .attr('font-size', '12px')
            .attr('dx', 12)
            .attr('dy', 4);

        // 시뮬레이션 업데이트
        simulation.nodes(graphData.nodes)
            .on('tick', ticked);

        simulation.force('link')
            .links(graphData.links);

        function ticked() {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);

            label
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        }

        function dragstarted(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }

        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }

        function dragended(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }

    } catch (error) {
        console.error('지식 그래프 로딩 실패:', error);
        document.getElementById('knowledge-graph').innerHTML =
            '<p style="color: red;">지식 그래프를 불러오는데 실패했습니다.</p>';
    }
}