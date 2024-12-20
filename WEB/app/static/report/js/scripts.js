document.addEventListener("DOMContentLoaded", function () {
    const calendarEl = document.getElementById("calendar");
    const calendar = new FullCalendar.Calendar(calendarEl, {
        initialView: "dayGridMonth",
        locale: "ko",
        headerToolbar: {
            left: "prev,next today",
            center: "title",
            right: "dayGridMonth,dayGridWeek,dayGridDay",
            // dayMaxEvents: 1, // 하루에 표시할 최대 이벤트 수
            eventDisplay: 'block', // 이벤트를 블록 형태로 표시
            height: 'auto', // 캘린더 높이 자동 조정
        },
        events: calendarEvents,

        eventClick: function (info) {
            fetchLLMReport(info.event.startStr);
        },
        eventDidMount: function(info) {
            console.log('Event props:', info.event.extendedProps); // 디버깅용

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

    span.onclick = function () {
        modal.style.display = "none";
    };

    window.onclick = function (event) {
        if (event.target == modal) {
            modal.style.display = "none";
        }
    };

    // fetchLLMReport 함수도 window에 등록
    window.fetchLLMReport = function (date) {
        const output = document.getElementById("llm-output");
        const additionalInfo = document.getElementById("additional-info");
        output.innerHTML = "불러오는 중...";
        additionalInfo.innerHTML = ""; // 기존의 추가 정보 초기화
        modal.style.display = "block";

        async function fetchReportStream() {
            try {
                const response = await fetch(`/api/streaming-daily-report/${userId}/?date=${date}`);
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

                // 추가 정보 처리 (예시로 추가)
                additionalInfo.innerHTML = "추가 정보: 예시 리포트 내용";

            } catch (error) {
                console.error("스트리밍 오류:", error);
                output.innerHTML = `<p style="color:red;">오류 발생: ${error.message}</p>`;
            }
        }

        function updateOutput(element, content) {
            requestAnimationFrame(() => {
                element.innerHTML = marked.parse(content);
            });
        }

        fetchReportStream();
    };

    // 초기 상태에서 챗봇을 닫아둠
    chatbot.style.display = "none";

    // 챗봇 열기/닫기 함수
    window.toggleChatbot = function () {
        if (chatbot.style.display === "none") {
            chatbot.style.display = "flex"; // 열기
        } else {
            chatbot.style.display = "none"; // 닫기
        }
    };

    // 메시지 전송 함수
    window.sendMessage = function () {
        const inputField = document.getElementById("chatbot-text");
        const message = inputField.value;
        if (message.trim() !== "") {
            const messageDiv = document.createElement("div");
            messageDiv.textContent = "나: " + message;
            document.getElementById("chatbot-messages").appendChild(messageDiv);
            inputField.value = "";
            setTimeout(() => {
                const responseDiv = document.createElement("div");
                responseDiv.textContent = "챗봇: 이건 예시 응답입니다.";
                document.getElementById("chatbot-messages").appendChild(responseDiv);
                document.getElementById("chatbot-messages").scrollTop = document.getElementById("chatbot-messages").scrollHeight;
            }, 1000);
        }
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
});
