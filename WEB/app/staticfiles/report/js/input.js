function validateForm() {
    var userId = document.getElementById("user_id").value;

    // 입력값이 비어있는 경우 경고 메시지 출력
    if (userId == "") {
        alert("User ID를 입력하세요.");
        return false;
    }

    // UUID 형식 검증 (간단한 정규식 사용)
    var uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
    if (!uuidRegex.test(userId)) {
        alert("잘못된 UUID 형식입니다. 올바른 형식을 입력하세요.");
        return false;
    }

    return true; // 모든 검증을 통과한 경우 폼 제출 허용
}
