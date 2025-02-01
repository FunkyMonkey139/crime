

const WebSocket = require('ws');
const axios = require('axios');

// Create a WebSocket server
const wss = new WebSocket.Server({ port: 8080 });

wss.on('connection', (ws) => {
    console.log("New WebSocket connection established.");

    // Function to fetch data from Flask server
    const fetchData = async () => {
        try {
            const response = await axios.get("http://127.0.0.1:5003/detect", {
                responseType: "stream",
            });

            response.data.on('data', (chunk) => {
                const data = chunk.toString();
                // Parse only valid JSON data and send it
                if (data.startsWith("data:")) {
                    const jsonData = data.replace("data:", "").trim();
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(jsonData);
                    }
                }
            });
        } catch (err) {
            console.error("Error fetching data from Flask server:", err.message);
        }
    };

    // Start fetching data
    fetchData();

    // Handle WebSocket close
    ws.on('close', () => {
        console.log("WebSocket connection closed.");
    });
});

console.log("WebSocket server running on ws://localhost:8080");
