/**
 * Emotion Drive — frontend logic
 * WebSocket telemetry, virtual arena preview, and control inputs.
 */

// Global State
let ws = null;
let currentMode = "AI";
let activeAction = "STOP";
let currentSpeed = 200;

// DOM Elements
const fpsVal = document.getElementById("fpsVal");
const latencyVal = document.getElementById("latencyVal");
const targetVal = document.getElementById("targetVal");
const modeVal = document.getElementById("modeVal");
const faceDetectTag = document.getElementById("faceDetectTag");
const activeActionTag = document.getElementById("activeActionTag");

// Motion buttons
const arrowForward = document.getElementById("arrowForward");
const arrowBackward = document.getElementById("arrowBackward");
const arrowLeft = document.getElementById("arrowLeft");
const arrowRight = document.getElementById("arrowRight");
const coreStop = document.getElementById("coreStop");

// Telemetry Stats
const statPackets = document.getElementById("statPackets");
const statSpeed = document.getElementById("statSpeed");
const statCmd = document.getElementById("statCmd");

// Controls
const btnModeAI = document.getElementById("btnModeAI");
const btnModeManual = document.getElementById("btnModeManual");
const emergencyStopBtn = document.getElementById("emergencyStopBtn");
const speedSlider = document.getElementById("speedSlider");
const speedDisplay = document.getElementById("speedDisplay");
const turnSpeedSlider = document.getElementById("turnSpeedSlider");
const turnSpeedDisplay = document.getElementById("turnSpeedDisplay");
const threshSlider = document.getElementById("threshSlider");
const threshDisplay = document.getElementById("threshDisplay");
const btnSaveNetwork = document.getElementById("btnSaveNetwork");
const targetIpInput = document.getElementById("targetIpInput");
const targetPortInput = document.getElementById("targetPortInput");

// Re-binders
const mapHappy = document.getElementById("mapHappy");
const mapSad = document.getElementById("mapSad");
const mapAngry = document.getElementById("mapAngry");
const mapSurprised = document.getElementById("mapSurprised");

// Virtual Arena Setup
const arenaCanvas = document.getElementById("arenaCanvas");
const ctx = arenaCanvas.getContext("2d");
const resetArenaBtn = document.getElementById("resetArenaBtn");

let botSim = {
  x: arenaCanvas.width / 2,
  y: arenaCanvas.height / 2,
  angle: -Math.PI / 2, // Facing up
  speed: 0,
  targetSpeed: 0,
  turnSpeed: 0,
  targetTurnSpeed: 0,
  trail: [],
};

// Keep arena crisp on high-DPI / responsive layouts.
// Logical sim space stays 640x360; backing store scales with displayed size.
function fitArenaCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = arenaCanvas.getBoundingClientRect();
  const cssW = Math.max(280, rect.width || 640);
  const cssH = cssW * (9 / 16);
  arenaCanvas.style.height = `${cssH}px`;
  const backingW = Math.round(cssW * dpr);
  const backingH = Math.round(cssH * dpr);
  if (arenaCanvas.width !== backingW || arenaCanvas.height !== backingH) {
    const sx = backingW / 640;
    const sy = backingH / 360;
    // Rescale sim position proportionally
    botSim.x = (botSim.x / 640) * backingW || backingW / 2;
    botSim.y = (botSim.y / 360) * backingH || backingH / 2;
    botSim.trail = botSim.trail.map((p) => ({ x: (p.x / 640) * backingW, y: (p.y / 360) * backingH }));
    arenaCanvas.width = backingW;
    arenaCanvas.height = backingH;
    ctx.setTransform(sx, 0, 0, sy, 0, 0);
  } else {
    ctx.setTransform(backingW / 640, 0, 0, backingH / 360, 0, 0);
  }
}

window.addEventListener("resize", fitArenaCanvas);

// ============================================================================
// 1. WEBSOCKET TELEMETRY
// ============================================================================
function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log("[WebSocket] Connected to Telemetry Stream");
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      updateTelemetryUI(data);
    } catch (e) {
      console.error("Telemetry parse error:", e);
    }
  };

  ws.onclose = () => {
    console.warn("[WebSocket] Disconnected. Reconnecting in 1s...");
    setTimeout(connectWebSocket, 1000);
  };
}

function updateTelemetryUI(data) {
  // Update header chips
  fpsVal.textContent = data.fps.toFixed(1);
  latencyVal.textContent = `${data.latency_ms.toFixed(0)} ms`;
  modeVal.textContent = data.mode === "AI" ? "AI Auto" : "Manual";
  currentMode = data.mode;

  if (data.mode === "AI") {
    btnModeAI.classList.add("active");
    btnModeManual.classList.remove("active");
  } else {
    btnModeManual.classList.add("active");
    btnModeAI.classList.remove("active");
  }

  // Face badge
  if (data.face_detected) {
    faceDetectTag.textContent = "Face detected";
    faceDetectTag.classList.add("locked");
  } else {
    faceDetectTag.textContent = "No face";
    faceDetectTag.classList.remove("locked");
  }

  // Active Action
  activeAction = data.active_action;
  activeActionTag.textContent = activeAction;

  // Emotion Probability Bars
  if (data.scores) {
    for (const [emo, score] of Object.entries(data.scores)) {
      const row = document.querySelector(`.gauge-row[data-emotion="${emo}"]`);
      if (row) {
        const fill = row.querySelector(".bar-fill");
        const pct = row.querySelector(".emo-pct");
        const val = Math.round(score * 100);
        if (fill) fill.style.width = `${val}%`;
        if (pct) pct.textContent = `${val}%`;

        if (data.active_emotion === emo) {
          row.style.opacity = "1";
        } else {
          row.style.opacity = "0.75";
        }
      }
    }
  }

  // Motion highlight
  arrowForward.classList.toggle("active", activeAction === "FORWARD");
  arrowBackward.classList.toggle("active", activeAction === "BACKWARD");
  arrowLeft.classList.toggle("active", activeAction === "LEFT");
  arrowRight.classList.toggle("active", activeAction === "RIGHT");
  coreStop.classList.toggle("active", activeAction === "STOP");

  // Link stats
  if (data.udp) {
    statPackets.textContent = data.udp.packets_sent;
    statSpeed.textContent = data.udp.speed;
    statCmd.textContent = data.udp.active_cmd;
    targetVal.textContent = `${data.udp.target_ip}:${data.udp.target_port}`;
  }

  // Sync turn speed slider if not actively being dragged
  if (data.turn_speed && turnSpeedSlider && turnSpeedDisplay && document.activeElement !== turnSpeedSlider) {
    turnSpeedSlider.value = data.turn_speed;
    turnSpeedDisplay.textContent = data.turn_speed;
  }
}

// ============================================================================
// 2. VIRTUAL ARENA SIMULATOR (minimal warm palette)
// ============================================================================
const ARENA_W = 640;
const ARENA_H = 360;

function updateVirtualBot() {
  const maxLinearSpeed = 3.2;
  const maxTurnSpeed = 0.028; // Slow and precise rotation speed

  // Set target speeds based on activeAction
  if (activeAction === "FORWARD") {
    botSim.targetSpeed = maxLinearSpeed;
    botSim.targetTurnSpeed = 0;
  } else if (activeAction === "BACKWARD") {
    botSim.targetSpeed = -maxLinearSpeed * 0.8;
    botSim.targetTurnSpeed = 0;
  } else if (activeAction === "LEFT") {
    botSim.targetSpeed = 0.0;
    botSim.targetTurnSpeed = -maxTurnSpeed;
  } else if (activeAction === "RIGHT") {
    botSim.targetSpeed = 0.0;
    botSim.targetTurnSpeed = maxTurnSpeed;
  } else {
    botSim.targetSpeed = 0;
    botSim.targetTurnSpeed = 0;
  }

  // Smooth acceleration
  botSim.speed += (botSim.targetSpeed - botSim.speed) * 0.2;
  botSim.turnSpeed += (botSim.targetTurnSpeed - botSim.turnSpeed) * 0.25;

  // Integrate position (logical 640x360 space)
  const scaleX = ARENA_W / arenaCanvas.width;
  const scaleY = ARENA_H / arenaCanvas.height;
  // Work in logical coordinates by converting current pixel pos
  let lx = (botSim.x / arenaCanvas.width) * ARENA_W;
  let ly = (botSim.y / arenaCanvas.height) * ARENA_H;

  botSim.angle += botSim.turnSpeed;
  lx += Math.cos(botSim.angle) * botSim.speed;
  ly += Math.sin(botSim.angle) * botSim.speed;

  // Clamp within arena
  const margin = 24;
  if (lx < margin) { lx = margin; botSim.angle = Math.PI - botSim.angle; }
  if (lx > ARENA_W - margin) { lx = ARENA_W - margin; botSim.angle = Math.PI - botSim.angle; }
  if (ly < margin) { ly = margin; botSim.angle = -botSim.angle; }
  if (ly > ARENA_H - margin) { ly = ARENA_H - margin; botSim.angle = -botSim.angle; }

  botSim.x = (lx / ARENA_W) * arenaCanvas.width;
  botSim.y = (ly / ARENA_H) * arenaCanvas.height;
  void scaleX; void scaleY;

  // Trail (store in device pixels, cap length)
  if (Math.abs(botSim.speed) > 0.15 || Math.abs(botSim.turnSpeed) > 0.01) {
    botSim.trail.push({ x: botSim.x, y: botSim.y });
    if (botSim.trail.length > 60) botSim.trail.shift();
  }
}

function drawVirtualArena() {
  // Draw in logical 640x360 coordinates (transform already set in fitArenaCanvas)
  ctx.save();
  ctx.setTransform(arenaCanvas.width / ARENA_W, 0, 0, arenaCanvas.height / ARENA_H, 0, 0);
  ctx.clearRect(0, 0, ARENA_W, ARENA_H);

  // Background
  ctx.fillStyle = "#faf9f6";
  ctx.fillRect(0, 0, ARENA_W, ARENA_H);

  // Grid — subtle warm gray
  ctx.strokeStyle = "rgba(28, 25, 23, 0.07)";
  ctx.lineWidth = 1;
  const gridSize = 32;
  for (let x = 0; x <= ARENA_W; x += gridSize) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, ARENA_H);
    ctx.stroke();
  }
  for (let y = 0; y <= ARENA_H; y += gridSize) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(ARENA_W, y);
    ctx.stroke();
  }

  // Convert trail back to logical coords for drawing
  if (botSim.trail.length > 1) {
    ctx.beginPath();
    const toLogical = (p) => ({
      x: (p.x / arenaCanvas.width) * ARENA_W,
      y: (p.y / arenaCanvas.height) * ARENA_H,
    });
    const first = toLogical(botSim.trail[0]);
    ctx.moveTo(first.x, first.y);
    for (let i = 1; i < botSim.trail.length; i++) {
      const p = toLogical(botSim.trail[i]);
      ctx.lineTo(p.x, p.y);
    }
    ctx.strokeStyle = "rgba(68, 64, 60, 0.45)";
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.stroke();
  }

  // Robot body in logical coords
  const lx = (botSim.x / arenaCanvas.width) * ARENA_W;
  const ly = (botSim.y / arenaCanvas.height) * ARENA_H;

  ctx.save();
  ctx.translate(lx, ly);
  ctx.rotate(botSim.angle);

  // Soft shadow
  ctx.fillStyle = "rgba(28, 25, 23, 0.12)";
  ctx.beginPath();
  ctx.ellipse(0, 3, 34, 24, 0, 0, Math.PI * 2);
  ctx.fill();

  // Wheels — muted stone
  ctx.fillStyle = "#78716c";
  ctx.fillRect(-20, -25, 17, 10);
  ctx.fillRect(-20, 15, 17, 10);

  // Chassis — white with dark outline
  ctx.fillStyle = "#ffffff";
  ctx.strokeStyle = "#1c1917";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  if (ctx.roundRect) {
    ctx.roundRect(-26, -19, 52, 38, 9);
  } else {
    ctx.rect(-26, -19, 52, 38);
  }
  ctx.fill();
  ctx.stroke();

  // Direction pointer — warm clay
  ctx.fillStyle = "#c2410c";
  ctx.beginPath();
  ctx.moveTo(21, 0);
  ctx.lineTo(5, -10);
  ctx.lineTo(5, 10);
  ctx.closePath();
  ctx.fill();

  // Center dot
  ctx.fillStyle = "#1c1917";
  ctx.beginPath();
  ctx.arc(-9, 0, 4.5, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
  ctx.restore();
}

function arenaLoop() {
  updateVirtualBot();
  drawVirtualArena();
  requestAnimationFrame(arenaLoop);
}

// ============================================================================
// 3. EVENT HANDLERS & API CALLS
// ============================================================================
async function sendManualCommand(action, speed = null) {
  try {
    await fetch("/api/manual_command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: action, speed: speed }),
    });
  } catch (e) {
    console.error("Failed to send manual command:", e);
  }
}

async function setMode(mode) {
  try {
    await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: mode }),
    });
  } catch (e) {
    console.error("Failed to update mode:", e);
  }
}

async function triggerEmergencyStop() {
  try {
    await fetch("/api/emergency_stop", { method: "POST" });
  } catch (e) {
    console.error("Failed emergency stop:", e);
  }
}

// Mode Buttons
btnModeAI.addEventListener("click", () => setMode("AI"));
btnModeManual.addEventListener("click", () => setMode("MANUAL"));
emergencyStopBtn.addEventListener("click", triggerEmergencyStop);

// Slider Listeners
speedSlider.addEventListener("input", (e) => {
  currentSpeed = parseInt(e.target.value);
  speedDisplay.textContent = currentSpeed;
});
speedSlider.addEventListener("change", async (e) => {
  await fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ speed: parseInt(e.target.value) }),
  });
});

if (turnSpeedSlider && turnSpeedDisplay) {
  turnSpeedSlider.addEventListener("input", (e) => {
    turnSpeedDisplay.textContent = e.target.value;
  });
  turnSpeedSlider.addEventListener("change", async (e) => {
    await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ turn_speed: parseInt(e.target.value) }),
    });
  });
}

threshSlider.addEventListener("input", (e) => {
  threshDisplay.textContent = `${e.target.value}%`;
});
threshSlider.addEventListener("change", async (e) => {
  const fraction = parseFloat(e.target.value) / 100.0;
  await fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ activation_threshold: fraction }),
  });
});

// Mapping Re-binder listeners
function updateMappings() {
  const mapping = {
    Happy: mapHappy.value,
    Sad: mapSad.value,
    Angry: mapAngry.value,
    Surprised: mapSurprised.value,
  };
  fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mapping: mapping }),
  });
}
mapHappy.addEventListener("change", updateMappings);
mapSad.addEventListener("change", updateMappings);
mapAngry.addEventListener("change", updateMappings);
mapSurprised.addEventListener("change", updateMappings);

// Network Config
btnSaveNetwork.addEventListener("click", async () => {
  const ip = targetIpInput.value.trim();
  const port = parseInt(targetPortInput.value);
  if (ip && port) {
    await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target_ip: ip, target_port: port }),
    });
  }
});

// Reset Arena
function resetArena() {
  botSim.x = arenaCanvas.width / 2;
  botSim.y = arenaCanvas.height / 2;
  botSim.angle = -Math.PI / 2;
  botSim.speed = 0;
  botSim.turnSpeed = 0;
  botSim.targetSpeed = 0;
  botSim.targetTurnSpeed = 0;
  botSim.trail = [];
}
resetArenaBtn.addEventListener("click", resetArena);

// D-Pad: pointer events cover mouse + touch
function bindHoldButton(id, action) {
  const el = document.getElementById(id);
  if (!el) return;
  const down = (e) => {
    e.preventDefault();
    el.classList.add("pressed");
    sendManualCommand(action);
  };
  const up = () => {
    el.classList.remove("pressed");
    if (currentMode === "MANUAL" && action !== "STOP") sendManualCommand("STOP");
  };
  el.addEventListener("pointerdown", down);
  el.addEventListener("pointerup", up);
  el.addEventListener("pointerleave", () => el.classList.remove("pressed"));
  el.addEventListener("pointercancel", up);
}

bindHoldButton("dpadUp", "FORWARD");
bindHoldButton("dpadDown", "BACKWARD");
bindHoldButton("dpadLeft", "LEFT");
bindHoldButton("dpadRight", "RIGHT");
bindHoldButton("dpadStop", "STOP");

// Keyboard Controls (WASD & Spacebar)
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

  if (e.code === "Space") {
    e.preventDefault();
    triggerEmergencyStop();
    return;
  }

  if (currentMode !== "MANUAL") return;
  if (e.repeat) return;

  if (e.key === "w" || e.key === "W" || e.key === "ArrowUp") {
    sendManualCommand("FORWARD");
  } else if (e.key === "s" || e.key === "S" || e.key === "ArrowDown") {
    sendManualCommand("BACKWARD");
  } else if (e.key === "a" || e.key === "A" || e.key === "ArrowLeft") {
    sendManualCommand("LEFT");
  } else if (e.key === "d" || e.key === "D" || e.key === "ArrowRight") {
    sendManualCommand("RIGHT");
  }
});

window.addEventListener("keyup", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (currentMode === "MANUAL") {
    if (["w", "s", "a", "d", "W", "S", "A", "D", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.key)) {
      sendManualCommand("STOP");
    }
  }
});

// Start WebSocket and Arena Loop
fitArenaCanvas();
connectWebSocket();
arenaLoop();
setTimeout(fitArenaCanvas, 100);
