async function sendMessage() {
    var userInput = document.getElementById("user-input").value.trim();

    if (userInput === "") {
        alert("Please enter a message.");
        return;
    }

    var chatLog = document.getElementById("chat-log");
    chatLog.innerHTML += `<p><strong>You:</strong> ${userInput}</p>`;

    document.getElementById("user-input").value = "";

    try {
        let response = await fetch("http://127.0.0.1:5000/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: userInput })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        let data = await response.json();
        let botResponse = data.response || "Sorry, an error occurred.";
        
        chatLog.innerHTML += `<p><strong>Chatbot:</strong> ${botResponse}</p>`;
        chatLog.scrollTop = chatLog.scrollHeight;
    } catch (error) {
        console.error("Error in fetching data:", error);
        chatLog.innerHTML += "<p><strong>Chatbot:</strong> Sorry, an error occurred.</p>";
    }
}

document.getElementById("user-input").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});