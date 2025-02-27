document.addEventListener("DOMContentLoaded", function () {
    const textarea = document.querySelector("textarea");

    function adjustHeight() {
        textarea.style.height = "auto";
        textarea.style.height = textarea.scrollHeight + "px"; 
    }

    textarea.addEventListener("input", adjustHeight);

    adjustHeight();
});
