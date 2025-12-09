import React from "react";
import "./App.css";
import LiveView from "./LiveView";

function App() {
  return (
    <div className="pm-app">
      {/* Top navigation bar */}
      <header className="pm-topbar">
        <div className="pm-topbar-left">
          <div className="pm-logo">
            <span className="pm-logo-icon">📷</span>
          </div>
          <div className="pm-logo-text">
            <div className="pm-logo-title">PhotoMentorAI</div>
            <div className="pm-logo-subtitle">Real-time Photography Assistant</div>
          </div>
        </div>

        <div className="pm-topbar-right">
          <button className="pm-pill pm-pill-active">AI Active</button>
          <div className="pm-user-chip">
            <div className="pm-avatar">RK</div>
            <div className="pm-user-meta">
              <div className="pm-user-name">Raju Kotturi</div>
              <div className="pm-user-role">Pro Mode</div>
            </div>
          </div>
        </div>
      </header>

      {/* Main content: live scene + AI mentor sidebar */}
      <main className="pm-main">
        <LiveView />
      </main>
    </div>
  );
}

export default App;