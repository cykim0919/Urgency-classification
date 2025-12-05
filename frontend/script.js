async function sendRequest() {
    const text = document.getElementById("userInput").value;

    const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
    });

    const data = await response.json();

    document.getElementById("resultBox").innerHTML =
        `긴급도: <b>${data.emergency_level}</b> (${data.meaning})`;
}
