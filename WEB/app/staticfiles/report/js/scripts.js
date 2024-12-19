document.addEventListener("DOMContentLoaded", function () {
    const calendarEl = document.getElementById("calendar");
    const calendar = new FullCalendar.Calendar(calendarEl, {
        initialView: "dayGridMonth",
        locale: "ko",
        headerToolbar: {
            left: "prev,next today",
            center: "title",
            right: "dayGridMonth,dayGridWeek,dayGridDay",
        },
        events: calendarEvents,
        eventClick: function (info) {
            fetchLLMReport(info.event.startStr);
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
        output.innerHTML = "불러오는 중...";
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

                output.innerHTML = marked.parse(accumulatedText);
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
});
