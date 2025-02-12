import React, { useState, useEffect } from "react";
import "./App.css";
import axios from 'axios';

const Navbar = () => {
  return (
    <nav className="navbar">
      <h1 className="header">LLM PenTester</h1>
      <div className="account-buttons">
        <Button text="Sign Up" primary />
        <Button text="Login" />
      </div>
    </nav>
  );
};

const TestingForm = ({ endpoint, setEndpoint, apiKey, setApiKey, setResult }) => {
  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log("Endpoint:", endpoint);
    console.log("API Key:", apiKey);
    const local_host = 'http://localhost:8080/predict';
    try {
      const response = await axios.post(local_host, {
        "endpoint": endpoint, "api_key": apiKey
      });
      if (response.data) {
        console.log(response.data);
        setResult(response.data);
      }
    } catch (error) {
      console.error('Error connecting localHost:', error);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <div className="testform">
        <h2 className="testform title">LLM Security Testing Platform</h2>
        <div className="testform input">
          <InputField label="LLM Endpoint URL" placeholder="https://api.example.com/v1/completions" value={endpoint} onChange={setEndpoint} />
          <InputField label="API Key" placeholder="Enter your API key" type="password" value={apiKey} onChange={setApiKey} />
          <button style={{
    background: "linear-gradient(to bottom, #36e4ff, #000dff)", // Gradient effect
    color: "white",
    borderRadius: "8px",
    padding: "12px 20px",
    border: "1px solid #8e44ad", 
    boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.2)", // Soft shadow
    position: "relative",
    overflow: "hidden",
  }} className="w-full py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transform hover:scale-105 transition-all flex items-center justify-center gap-2" type="submit">
            <span className="material-symbols-outlined">security</span> Start Penetration Testing
          </button>
        </div>
      </div>
    </form>
  );
};

const InputField = ({ label, placeholder, type = "text", value, onChange }) => {
  return (
    <div className="space-y-2">
      <label className="custom-label">{label}</label>
      <input
        type={type}
        className="w-full px-4 py-2 rounded-lg border focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none transition-all"
        placeholder={placeholder}
        value={value || ""}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  );
};

const Button = ({ text, primary }) => {
  return (
    <button
      className={`px-6 py-2 rounded-lg ${primary ? "bg-purple-600 text-white hover:bg-purple-700" : "border border-purple-600 hover:bg-purple-100"} transform hover:scale-105 transition-all w-20`}
    >
      {text}
    </button>
  );
};

const DownloadButton = ({ fileUrl, file_name }) => {
  console.log(`${fileUrl}/${file_name}`);
  let path = `public/output/${fileUrl}/${file_name}`;

  const getButtonLabel = (file_name) => {
    if (file_name === "llm_vulnerability_report.csv") return "Download CSV Report";
    if (file_name === "llm_penetration_report.pdf") return "Download PDF Report";
    return "Download File";
  };

  return (
    <div className="download-button">
      <a href={path} download className="download-button">
        <button className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 cursor-pointer">
          {getButtonLabel(file_name)}
        </button>
      </a>
    </div>
  );
};

function App() {
  const [result, setResult] = useState(null);
  const [endpoint, setEndpoint] = useState("");
  const [apiKey, setApiKey] = useState("");

  useEffect(() => {
    if (result !== null) {
      console.log('Updated result:', result);
    }
  }, [result]);

  return (
    <div className="llm-pentester">
      <video autoPlay loop muted className="video-bg">
        <source src="/background.mp4" type="video/mp4" />
      </video>
      <Navbar />
      <main className="p-8 flex justify-center">
        <TestingForm endpoint={endpoint} setEndpoint={setEndpoint} apiKey={apiKey} setApiKey={setApiKey} setResult={setResult} />
      </main>
      {result &&
        <div className="result">
          <DownloadButton fileUrl={result["folder_name"]} file_name="llm_vulnerability_report.csv" />
          <DownloadButton fileUrl={result["folder_name"]} file_name="llm_penetration_report.pdf" />
        </div>}
    </div>
  );
}

export default App;
