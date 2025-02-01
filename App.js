// import './App.css';
// import { useEffect, useState, useRef } from 'react';
// import Webcam from "react-webcam";
// import { Button, Card, Container, Row, Col } from 'react-bootstrap';  // Using react-bootstrap for better UI

// function App() {
//   const [data, setData] = useState("");
//   const webcamRef = useRef(null); // Define webcamRef using useRef

//   const getData = () => {
//     // Create a WebSocket connection
//     const socket = new WebSocket("ws://localhost:8080");

//     // Handle incoming messages
//     socket.onmessage = (event) => {
//       setData(event.data); // Update the state with the received message
//     };

//     // Handle WebSocket errors
//     socket.onerror = (error) => {
//       console.error("WebSocket error:", error);
//     };

//     // Handle WebSocket closure
//     socket.onclose = () => {
//       console.log("WebSocket connection closed");
//     };
//   };

//   useEffect(() => {
//     getData();
//   }, []);

//   const capture = () => {
//     if (webcamRef.current) {
//       const screenshot = webcamRef.current.getScreenshot();
//       console.log("Captured Screenshot:", screenshot);
//     }
//   };

//   return (
//     <Container fluid className="app-container">
//       <Row className="header-row">
//         <Col>
//           <h1 className="title">Camera Feed</h1>
//         </Col>
//       </Row>

//       <Row>
//         <Col md={6} className="webcam-col">
//           <Card className="webcam-card">
//             <Webcam
//               audio={false}
//               ref={webcamRef} // Attach the ref to the Webcam component
//               screenshotFormat="image/jpeg"
//               width="100%" // Make webcam responsive
//               videoConstraints={{ facingMode: "environment" }} // Prefer rear camera
//             />
//           </Card>
//           <Button variant="primary" onClick={capture} className="capture-btn">
//             Capture Photo
//           </Button>
//         </Col>

//         <Col md={6} className="data-col">
//           <Card className="data-card">
//             <Card.Body>
//               <h2 className="data-title">WebSocket Data:</h2>
//               <div className="data-content">{data}</div>
//             </Card.Body>
//           </Card>
//         </Col>
//       </Row>
//     </Container>
//   );
// }

// export default App;

// import './App.css';
// import { useEffect, useState, useRef } from 'react';
// import Webcam from "react-webcam";
// import { Button, Card, Container, Row, Col, Spinner } from 'react-bootstrap';  // Using react-bootstrap for better UI

// function App() {
//   const [data, setData] = useState(null);
//   const [loading, setLoading] = useState(true); // State to handle loading status
//   const [screenshot, setScreenshot] = useState(null); // State to store captured screenshot
//   const webcamRef = useRef(null); // Define webcamRef using useRef

//   const getData = () => {
//     // Create a WebSocket connection
//     const socket = new WebSocket("ws://localhost:8080");

//     // Handle incoming messages
//     socket.onmessage = (event) => {
//       setData(JSON.parse(event.data)); // Update the state with the received message
//       setLoading(false); // Set loading to false once data is received
//     };

//     // Handle WebSocket errors
//     socket.onerror = (error) => {
//       console.error("WebSocket error:", error);
//       setLoading(false);
//     };

//     // Handle WebSocket closure
//     socket.onclose = () => {
//       console.log("WebSocket connection closed");
//     };
//   };

//   useEffect(() => {
//     getData();
//   }, []);

//   const capture = () => {
//     if (webcamRef.current) {
//       const screenshot = webcamRef.current.getScreenshot();
//       setScreenshot(screenshot); // Save the screenshot to state
//       console.log("Captured Screenshot:", screenshot);
//     }
//   };

//   return (
//     <Container fluid className="app-container">
//       <Row className="header-row">
//         <Col>
//           <h1 className="title">Camera Feed</h1>
//         </Col>
//       </Row>

//       <Row>
//         <Col md={6} className="webcam-col">
//           <Card className="webcam-card">
//             <Webcam
//               audio={false}
//               ref={webcamRef} // Attach the ref to the Webcam component
//               screenshotFormat="image/jpeg"
//               width="100%" // Make webcam responsive
//               videoConstraints={{ facingMode: "environment" }} // Prefer rear camera
//             />
//           </Card>
//           <Button variant="primary" onClick={capture} className="capture-btn">
//             Capture Photo
//           </Button>

//           {screenshot && (
//             <div className="screenshot-preview">
//               <h3>Captured Screenshot:</h3>
//               <img src={screenshot} alt="Captured" className="screenshot-img" />
//             </div>
//           )}
//         </Col>

//         <Col md={6} className="data-col">
//           <Card className="data-card">
//             <Card.Body>
//               <h2 className="data-title">WebSocket Data:</h2>
//               {loading ? (
//                 <Spinner animation="border" variant="primary" />
//               ) : (
//                 <pre className="data-content">{JSON.stringify(data, null, 2)}</pre> // Pretty print JSON data
//               )}
//             </Card.Body>
//           </Card>
//         </Col>
//       </Row>
//     </Container>
//   );
// }

// export default App;
import './App.css';
import { useEffect, useState, useRef } from 'react';
import Webcam from "react-webcam";
import { Button, Card, Container, Row, Col, Spinner } from 'react-bootstrap';  // Using react-bootstrap for better UI

function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true); // State to handle loading status
  const [screenshot, setScreenshot] = useState(null); // State to store captured screenshot
  const webcamRef = useRef(null); // Define webcamRef using useRef

  const getData = () => {
    // Create a WebSocket connection
    const socket = new WebSocket("ws://localhost:8080");

    // Handle incoming messages
    socket.onmessage = (event) => {
      setData(JSON.parse(event.data)); // Update the state with the received message
      setLoading(false); // Set loading to false once data is received
    };

    // Handle WebSocket errors
    socket.onerror = (error) => {
      console.error("WebSocket error:", error);
      setLoading(false);
    };

    // Handle WebSocket closure
    socket.onclose = () => {
      console.log("WebSocket connection closed");
    };
  };

  useEffect(() => {
    getData();
  }, []);

  const capture = () => {
    if (webcamRef.current) {
      const screenshot = webcamRef.current.getScreenshot();
      setScreenshot(screenshot); // Save the screenshot to state
      console.log("Captured Screenshot:", screenshot);
    }
  };

  return (
    <Container fluid className="app-container">
      <Row className="header-row">
        <Col>
          <h1 className="title">Camera Feed</h1>
        </Col>
      </Row>

      <Row>
        <Col md={6} className="webcam-col">
          <Card className="webcam-card">
            <Webcam
              audio={false}
              ref={webcamRef} // Attach the ref to the Webcam component
              screenshotFormat="image/jpeg"
              width="100%" // Make webcam responsive
              videoConstraints={{ facingMode: "environment" }} // Prefer rear camera
            />
          </Card>
          <Button variant="primary" onClick={capture} className="capture-btn">
            Capture Photo
          </Button>

          {screenshot && (
            <div className="screenshot-preview">
              <h3>Captured Screenshot:</h3>
              <img src={screenshot} alt="Captured" className="screenshot-img" />
            </div>
          )}
        </Col>

        <Col md={6} className="data-col">
          <Card className="data-card">
            <Card.Body>
              <h2 className="data-title">WebSocket Data:</h2>
              {loading ? (
                <Spinner animation="border" variant="primary" />
              ) : (
                <pre className="data-con1tent">{JSON.stringify(data, null, 2)}</pre> // Pretty print JSON data
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
}

export default App;