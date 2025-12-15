import React, { useEffect, useRef, useState, useCallback } from "react";

const BACKEND_URL = "https://photo-mentor-ai.onrender.com/analyze_frame";

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function mapExpressionToText(label) {
  switch (label) {
    case "happy":
      return "Happy 😊";
    case "neutral":
      return "Neutral 😐";
    case "serious":
      return "Serious / Focused 😶";
    case "eyes_closed":
      return "Eyes closed / obscured 😴";
    case "no_face":
      return "No face detected";
    case "unknown":
    default:
      return "Unknown";
  }
}

/**
 * Turn a long backend hint into a short spoken cue
 * when speechMode === "short".
 */
function normalizeHintForSpeech(text, category = "timing", speechMode = "full") {
  if (!text) return "";

  // In full mode, speak the whole sentence from backend.
  if (speechMode === "full") {
    return text;
  }

  const lower = text.toLowerCase();

  if (category === "timing") {
    if (lower.includes("capture now")) return "Capture now.";
    if (lower.includes("great expression")) return "Great expression. Capture now.";
    if (lower.includes("too dark")) return "Too dark.";
    if (lower.includes("too bright")) return "Too bright.";
    if (lower.includes("eyes may be closed")) return "Eyes closed. Wait.";
    if (lower.includes("neutral")) return "Neutral expression.";
    return "Adjust and capture.";
  }

  if (category === "exposure") {
    if (lower.includes("too dark")) return "Scene is too dark.";
    if (lower.includes("too bright")) return "Scene is too bright.";
    return "Check your exposure.";
  }

  if (category === "composition") {
    if (lower.includes("center")) return "Center your subject.";
    if (lower.includes("space above")) return "Lower the frame.";
    if (lower.includes("lower the frame")) return "Lower the frame.";
    if (lower.includes("raise the frame")) return "Raise the frame.";
    return "Adjust composition.";
  }

  // fallback
  return text;
}

function LiveView() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null); // keep track of current MediaStream

  // ✅ NEW: prevent request backlog
  const inFlightRef = useRef(false);

  const [hints, setHints] = useState({
    exposure_hint: "Waiting for camera…",
    face_hint: "",
    composition_hint: "",
    num_faces: 0,
    brightness: 0,
    timing_hint: "Waiting for subject…",
    timing_score: 40,
    expression_label: "no_face",
    expression_confidence: 0,
  });

  const [shots, setShots] = useState([]);
  const [showReview, setShowReview] = useState(false);

  // --- speech controls ---
  const [speechEnabled, setSpeechEnabled] = useState(true);
  const [speechMode, setSpeechMode] = useState("full"); // "full" | "short"
  const lastSpokenHintRef = useRef("");

  // --- camera side: "user" (front) or "environment" (back) ---
  const [cameraFacing, setCameraFacing] = useState("user");

  // Load speech preferences from localStorage
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const storedEnabled = window.localStorage.getItem("pm_speech_enabled");
      const storedMode = window.localStorage.getItem("pm_speech_mode");
      const storedFacing = window.localStorage.getItem("pm_camera_facing");

      if (storedEnabled !== null) {
        setSpeechEnabled(storedEnabled === "true");
      }
      if (storedMode === "full" || storedMode === "short") {
        setSpeechMode(storedMode);
      }
      if (storedFacing === "user" || storedFacing === "environment") {
        setCameraFacing(storedFacing);
      }
    } catch (e) {
      console.warn("Could not read preferences:", e);
    }
  }, []);

  // Persist preferences
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      window.localStorage.setItem("pm_speech_enabled", String(speechEnabled));
      window.localStorage.setItem("pm_speech_mode", speechMode);
      window.localStorage.setItem("pm_camera_facing", cameraFacing);
    } catch (e) {
      console.warn("Could not write preferences:", e);
    }
  }, [speechEnabled, speechMode, cameraFacing]);

  // Speak helper (memoized)
  const speakHint = useCallback(
    (text, category = "timing") => {
      if (!speechEnabled) return;
      if (typeof window === "undefined") return;

      const synth = window.speechSynthesis;
      if (!synth || !text) return;

      const spokenText = normalizeHintForSpeech(text, category, speechMode);
      if (!spokenText) return;

      // Stop any previous speech so new hint is clear
      synth.cancel();

      const utter = new SpeechSynthesisUtterance(spokenText);
      utter.lang = "en-US";
      utter.rate = speechMode === "short" ? 1.15 : 1.02;
      utter.pitch = 1.0;
      synth.speak(utter);
    },
    [speechEnabled, speechMode]
  );

  // Start / restart webcam when cameraFacing changes
  useEffect(() => {
    let localStream = null;

    const startCamera = async () => {
      try {
        // Stop any existing stream first
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((t) => t.stop());
          streamRef.current = null;
        }

        // Request new stream with facingMode
        const constraints = {
          video: {
            facingMode: cameraFacing, // "user" or "environment"
          },
        };

        localStream = await navigator.mediaDevices.getUserMedia(constraints);

        streamRef.current = localStream;

        if (videoRef.current) {
          videoRef.current.srcObject = localStream;
        }
      } catch (err) {
        console.error("Error accessing webcam:", err);
        setHints((prev) => ({
          ...prev,
          exposure_hint: "Unable to access webcam. Check browser camera permissions.",
        }));
      }
    };

    startCamera();

    return () => {
      // cleanup when effect re-runs or component unmounts
      if (localStream) {
        localStream.getTracks().forEach((t) => t.stop());
      }
      if (typeof window !== "undefined" && window.speechSynthesis) {
        window.speechSynthesis.cancel();
      }
    };
  }, [cameraFacing]);

  // Periodically send frames to backend (✅ faster + safer)
  useEffect(() => {
    let intervalId;

    const sendFrame = async () => {
      // ✅ prevent overlapping requests (no backlog)
      if (inFlightRef.current) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (!video || !canvas) return;
      if (video.readyState !== 4) return;

      const width = video.videoWidth;
      const height = video.videoHeight;
      if (!width || !height) return;

      // ✅ downscale before sending (big speed win)
      const TARGET_W = 480;
      const scale = width > TARGET_W ? TARGET_W / width : 1;
      const targetW = Math.round(width * scale);
      const targetH = Math.round(height * scale);

      canvas.width = targetW;
      canvas.height = targetH;

      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, targetW, targetH);

      // slightly lower quality → smaller payload
      const dataUrl = canvas.toDataURL("image/jpeg", 0.6);

      // ✅ add timeout so requests never hang
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 9000);

      inFlightRef.current = true;
      try {
        const res = await fetch(BACKEND_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_base64: dataUrl }),
          signal: controller.signal,
        });

        const json = await res.json();
        setHints((prev) => ({
          ...prev,
          ...json,
        }));
      } catch (err) {
        console.error("Error sending frame to backend:", err);
      } finally {
        clearTimeout(timeoutId);
        inFlightRef.current = false;
      }
    };

    // ✅ slower polling for cloud deployment (less lag)
    intervalId = setInterval(sendFrame, 1500);

    return () => clearInterval(intervalId);
  }, []);

  // SPEECH EFFECT: speak when the ACTION timing hint changes
  useEffect(() => {
    const hint = hints.timing_hint;

    if (!hint) return;
    if (typeof window === "undefined") return;

    // don't repeat same text over and over
    if (hint === lastSpokenHintRef.current) return;

    // only speak when we have a subject
    if (hints.num_faces === 0) return;

    // skip generic "Analyzing scene…" type messages
    if (hint.toLowerCase().includes("analyzing scene")) return;

    speakHint(hint, "timing");
    lastSpokenHintRef.current = hint;
  }, [hints.timing_hint, hints.num_faces, speakHint]);

  const lightingScore = clamp(Math.round((hints.brightness / 255) * 120), 5, 100);
  const compositionScore = hints.num_faces > 0 ? 90 : 60;
  const timingScore = clamp(hints.timing_score ?? 70, 0, 100);
  const settingsScore = 80;

  const expressionText = mapExpressionToText(hints.expression_label);

  const handleCapture = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) return;

    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, width, height);

    const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
    setShots((prev) => [...prev, dataUrl]);
  };

  const handleDeleteShot = (index) => {
    setShots((prev) => prev.filter((_, i) => i !== index));
  };

  const handleToggleSpeech = () => {
    const next = !speechEnabled;
    setSpeechEnabled(next);
    if (!next && typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
  };

  const handleFlipCamera = () => {
    setCameraFacing((prev) => (prev === "user" ? "environment" : "user"));
  };

  return (
    <>
      <div className="pm-layout">
        {/* LEFT: Live scene */}
        <section className="pm-scene">
          <div className="pm-scene-header">
            <span className="pm-scene-status-dot" />
            <span className="pm-scene-status-text">Analyzing scene…</span>
            <span className="pm-scene-live-pill">LIVE</span>
          </div>

          <div className="pm-scene-frame">
            <div className="pm-scene-grid" />
            <video ref={videoRef} autoPlay playsInline className="pm-scene-video" />

            <div className="pm-scene-bottom-bar">
              <div className="pm-scene-meta">Faces detected: {hints.num_faces || 0}</div>
            </div>
          </div>
        </section>

        {/* RIGHT: AI mentor sidebar */}
        <aside className="pm-sidebar">
          <header className="pm-sidebar-header">
            <div>
              <div className="pm-sidebar-title">AI Mentor</div>
              <div className="pm-sidebar-subtitle">Real-time guidance active</div>
            </div>
            <div className="pm-sidebar-actions">
              <button
                className="pm-icon-button"
                title={cameraFacing === "user" ? "Switch to back camera" : "Switch to front camera"}
                onClick={handleFlipCamera}
              >
                🔁
              </button>
              <button
                className="pm-icon-button"
                title={speechEnabled ? "Mute voice hints" : "Unmute voice hints"}
                onClick={handleToggleSpeech}
              >
                {speechEnabled ? "🔊" : "🔇"}
              </button>
            </div>
          </header>

          <section className="pm-card">
            <div className="pm-card-title">Scene Analysis</div>

            {/* Lighting: GREEN */}
            <div className="pm-meter-row">
              <div className="pm-meter-label">Lighting</div>
              <div className="pm-meter">
                <div className="pm-meter-fill pm-meter-fill-green" style={{ width: `${lightingScore}%` }} />
              </div>
              <div className="pm-meter-value">{lightingScore}%</div>
            </div>

            {/* Composition: AMBER */}
            <div className="pm-meter-row">
              <div className="pm-meter-label">Composition</div>
              <div className="pm-meter">
                <div className="pm-meter-fill pm-meter-fill-amber" style={{ width: `${compositionScore}%` }} />
              </div>
              <div className="pm-meter-value">{compositionScore}%</div>
            </div>

            {/* Timing: BLUE */}
            <div className="pm-meter-row">
              <div className="pm-meter-label">Timing</div>
              <div className="pm-meter">
                <div className="pm-meter-fill pm-meter-fill-blue" style={{ width: `${timingScore}%` }} />
              </div>
              <div className="pm-meter-value">{timingScore}%</div>
            </div>

            {/* Settings: PURPLE */}
            <div className="pm-meter-row">
              <div className="pm-meter-label">Settings</div>
              <div className="pm-meter">
                <div className="pm-meter-fill pm-meter-fill-purple" style={{ width: `${settingsScore}%` }} />
              </div>
              <div className="pm-meter-value">{settingsScore}%</div>
            </div>
          </section>

          <section className="pm-card pm-card-compact">
            <div className="pm-card-title-row">
              <span className="pm-card-title">Recommended Settings</span>
              <span className="pm-badge pm-badge-amber">AI Recommended</span>
            </div>

            <div className="pm-settings-row">
              <div className="pm-setting">
                <div className="pm-setting-label">F</div>
                <div className="pm-setting-value">2.8</div>
              </div>
              <div className="pm-setting">
                <div className="pm-setting-label">S</div>
                <div className="pm-setting-value">1/250</div>
              </div>
              <div className="pm-setting">
                <div className="pm-setting-label">ISO</div>
                <div className="pm-setting-value">400</div>
              </div>
              <div className="pm-setting">
                <div className="pm-setting-label">EV</div>
                <div className="pm-setting-value">+0.3</div>
              </div>
            </div>
          </section>

          <section className="pm-card">
            <div className="pm-card-title">Live Suggestions</div>

            <div className="pm-suggestion pm-suggestion-focus">
              <div className="pm-suggestion-label">FOCUS</div>
              <div className="pm-suggestion-text">
                {hints.face_hint || "Point the camera towards your subject."}
              </div>
            </div>

            <div className="pm-suggestion pm-suggestion-action">
              <div className="pm-suggestion-label">ACTION</div>
              <div className="pm-suggestion-text">
                {hints.timing_hint ||
                  hints.exposure_hint ||
                  "Watch for the moment with the best expression, then capture."}
              </div>
              <div className="pm-suggestion-expression">
                Expression:{" "}
                <span className="pm-suggestion-expression-value">
                  {expressionText}
                  {hints.expression_confidence ? ` · ${hints.expression_confidence}%` : ""}
                </span>
              </div>
            </div>

            <div className="pm-suggestion pm-suggestion-rotation">
              <div className="pm-suggestion-label">COMPOSITION</div>
              <div className="pm-suggestion-text">
                {hints.composition_hint || "Try aligning your subject on the rule-of-thirds grid."}
              </div>
            </div>
          </section>

          {/* Speech settings card */}
          <section className="pm-card pm-card-compact">
            <div className="pm-card-title">Speech Settings</div>
            <div className="pm-settings-row">
              <div className="pm-setting" style={{ width: "100%" }}>
                <div className="pm-setting-label">Mode</div>
                <select
                  className="pm-setting-select"
                  value={speechMode}
                  onChange={(e) => setSpeechMode(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "0.35rem 0.5rem",
                    borderRadius: "0.5rem",
                    border: "1px solid rgba(148, 163, 184, 0.6)",
                    background: "rgba(15, 23, 42, 0.4)",
                    color: "inherit",
                  }}
                >
                  <option value="full">Full guidance</option>
                  <option value="short">Short cues only</option>
                </select>
              </div>
            </div>
          </section>

          <section className="pm-footer-row">
            <button className="pm-primary-btn" onClick={handleCapture}>
              Capture
            </button>
            <button className="pm-ghost-btn" onClick={() => setShowReview(true)} disabled={shots.length === 0}>
              Review Shots ({shots.length})
            </button>
          </section>
        </aside>
      </div>

      <canvas ref={canvasRef} style={{ display: "none" }} />

      {showReview && (
        <div className="pm-review-backdrop" onClick={() => setShowReview(false)}>
          <div className="pm-review-modal" onClick={(e) => e.stopPropagation()}>
            <div className="pm-review-header">
              <h2>Your Captured Shots</h2>
              <button className="pm-icon-button" onClick={() => setShowReview(false)}>
                ✕
              </button>
            </div>

            {shots.length === 0 ? (
              <div className="pm-review-empty">No shots captured yet.</div>
            ) : (
              <div className="pm-review-grid">
                {shots.map((shot, idx) => (
                  <div className="pm-review-card" key={idx}>
                    <img src={shot} alt={`shot-${idx}`} />
                    <div className="pm-review-card-actions">
                      <a href={shot} download={`photo-${idx + 1}.jpg`}>
                        Download
                      </a>
                      <button onClick={() => handleDeleteShot(idx)}>Delete</button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}

export default LiveView;