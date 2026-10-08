let ws = null;
let reconnectTimer = null;
let totalLoggedEvents = 0;

function initWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log("Connected to telemetry WebSocket.");
    if (reconnectTimer) {
      clearInterval(reconnectTimer);
      reconnectTimer = null;
    }
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "telemetry") {
        updateDashboard(data);
      }
    } catch (e) {
      console.error("Error parsing telemetry message:", e);
    }
  };

  ws.onclose = () => {
    console.warn("WebSocket disconnected. Retrying in 2 seconds...");
    if (!reconnectTimer) {
      reconnectTimer = setInterval(initWebSocket, 2000);
    }
  };
}

function updateDashboard(payload) {
  const gameState = payload.game_state || {};
  const poses = payload.player_poses || {};
  const newEvents = payload.new_events || [];

  // 1. Update Match Status Badge & Timers
  const stateBadge = document.getElementById("matchStateBadge");
  const stateName = gameState.match_state || "LOBBY";
  stateBadge.innerText = stateName;
  stateBadge.className = `match-status-badge status-${stateName.toLowerCase()}`;

  const roundCounter = document.getElementById("roundCounter");
  const curRound = gameState.current_round || 0;
  const totRound = gameState.total_rounds || 40;
  roundCounter.innerText = `ROUND: ${curRound} / ${totRound}`;

  const timerDisplay = document.getElementById("timerDisplay");
  const activeSec = Math.floor(gameState.active_play_time || 0);
  const mins = String(Math.floor(activeSec / 60)).padStart(2, "0");
  const secs = String(activeSec % 60).padStart(2, "0");
  timerDisplay.innerText = `${mins}:${secs}`;

  // 2. Update 6-Player Cards
  const leaderboard = gameState.leaderboard || [];
  leaderboard.forEach((player) => {
    const pid = player.player_id;
    const pose = poses[pid] || {};

    // Rank Badge
    const rankEl = document.getElementById(`pRank${pid}`);
    if (rankEl) {
      rankEl.innerText = player.rank;
      rankEl.className = `rank-badge rank-${Math.min(3, player.rank)}`;
    }

    // Name
    const nameEl = document.getElementById(`pName${pid}`);
    if (nameEl) nameEl.innerText = player.name;

    // Penalty Pill
    const penEl = document.getElementById(`pPenalty${pid}`);
    if (penEl) penEl.innerText = `${player.penalty_score.toFixed(1)} pts`;

    // Gaze State Pill
    const gazeEl = document.getElementById(`pGazePill${pid}`);
    if (gazeEl) {
      const gState = pose.gaze_state || "FACE_NOT_DETECTED";
      gazeEl.innerText = gState;
      gazeEl.className = `gaze-status-pill ${getGazePillClass(gState)}`;
    }

    // Attention Bar
    const attEl = document.getElementById(`pAttention${pid}`);
    const attBar = document.getElementById(`pAttentionBar${pid}`);
    if (attEl) attEl.innerText = `${player.tv_attention_pct.toFixed(1)}%`;
    if (attBar) attBar.style.width = `${player.tv_attention_pct}%`;

    // Streaks & Look aways
    const streakEl = document.getElementById(`pStreak${pid}`);
    if (streakEl) streakEl.innerText = `${player.current_look_streak_sec.toFixed(1)}s`;

    const lookAwayEl = document.getElementById(`pLookAways${pid}`);
    if (lookAwayEl) lookAwayEl.innerText = player.look_away_count;

    // Euler angles
    const yawEl = document.getElementById(`pYaw${pid}`);
    const pitchEl = document.getElementById(`pPitch${pid}`);
    const rollEl = document.getElementById(`pRoll${pid}`);
    if (yawEl) yawEl.innerText = `${(pose.yaw || 0) >= 0 ? '+' : ''}${(pose.yaw || 0).toFixed(1)}°`;
    if (pitchEl) pitchEl.innerText = `${(pose.pitch || 0) >= 0 ? '+' : ''}${(pose.pitch || 0).toFixed(1)}°`;
    if (rollEl) rollEl.innerText = `${(pose.roll || 0) >= 0 ? '+' : ''}${(pose.roll || 0).toFixed(1)}°`;
  });

  // 3. Update Sidebar Leaderboard
  const lbList = document.getElementById("leaderboardList");
  if (lbList && leaderboard.length > 0) {
    lbList.innerHTML = leaderboard.map(p => `
      <div class="leaderboard-item">
        <div style="display:flex; align-items:center; gap:8px;">
          <div class="rank-badge rank-${Math.min(3, p.rank)}">${p.rank}</div>
          <span style="font-weight:700; font-size:0.85rem;">${p.name}</span>
        </div>
        <div style="text-align:right;">
          <div style="font-weight:800; font-size:0.9rem; color:#f87171;">${p.penalty_score.toFixed(1)}</div>
          <div style="font-size:0.7rem; color:#10b981;">${p.tv_attention_pct.toFixed(0)}% Attn</div>
        </div>
      </div>
    `).join("");
  }

  // 4. Append New Reaction Events to Ticker
  if (newEvents && newEvents.length > 0) {
    const ticker = document.getElementById("eventsTicker");
    newEvents.forEach(ev => {
      totalLoggedEvents++;
      const evDiv = document.createElement("div");
      evDiv.className = `event-item ${getEventCssClass(ev.event_type)}`;
      evDiv.innerHTML = `
        <div style="display:flex; justify-content:space-between; font-weight:700;">
          <span>Player ${ev.player_id}: ${formatEventType(ev.event_type)}</span>
          <span style="color:#f59e0b;">Int: ${(ev.intensity * 100).toFixed(0)}%</span>
        </div>
        <div style="color:#cbd5e1; font-size:0.74rem;">${ev.description}</div>
        <div class="event-meta">
          <span>Game Time: ${ev.game_time_sec.toFixed(1)}s</span>
          <span>Round #${ev.video_round}</span>
        </div>
      `;
      // Insert at top
      if (ticker.firstChild) {
        ticker.insertBefore(evDiv, ticker.firstChild);
      } else {
        ticker.appendChild(evDiv);
      }
    });

    // Update count badge
    const badge = document.getElementById("eventCountBadge");
    if (badge) badge.innerText = totalLoggedEvents;
  }
}

function getGazePillClass(state) {
  switch (state) {
    case "FACING_TV": return "pill-facing-tv";
    case "LOOKING_DOWN": return "pill-looking-down";
    case "LOOKING_SIDEWAYS": return "pill-looking-sideways";
    case "FACING_AWAY": return "pill-facing-away";
    default: return "pill-lost";
  }
}

function getEventCssClass(type) {
  if (type.includes("HEAD")) return "head";
  if (type.includes("FACIAL")) return "facial";
  if (type.includes("POSTURE")) return "posture";
  if (type.includes("LOOK_AWAY")) return "away";
  return "disappeared";
}

function formatEventType(type) {
  return type.replace(/_/g, " ");
}

/* Control Actions */
async function triggerStart() {
  await fetch("/api/start", { method: "POST" });
}

async function triggerStop() {
  await fetch("/api/stop", { method: "POST" });
}

async function triggerCalibrate() {
  await fetch("/api/calibrate", { method: "POST" });
}

async function triggerSkip() {
  await fetch("/api/skip_video", { method: "POST" });
}

async function triggerReset() {
  if (confirm("Reset match session and clear all scores?")) {
    await fetch("/api/reset", { method: "POST" });
  }
}

async function onSensitivityChange(preset) {
  try {
    await fetch("/api/sensitivity", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ preset })
    });
    console.log("Sensitivity mode updated to:", preset);
  } catch (e) {
    console.error("Failed to update sensitivity:", e);
  }
}

/* ADB Modal */
async function openAdbModal() {
  document.getElementById("adbModal").style.display = "flex";
  refreshAdbStatus();
}

function closeAdbModal() {
  document.getElementById("adbModal").style.display = "none";
}

async function refreshAdbStatus() {
  const box = document.getElementById("adbDetails");
  box.innerText = "Querying connected ADB devices...";
  try {
    const res = await fetch("/api/adb/status");
    const data = await res.json();
    let text = `ADB Available: ${data.adb_available}\n`;
    text += `Binary: ${data.adb_path || 'Not installed'}\n\n`;
    text += `Connected Devices (${data.connected_devices.length}):\n`;
    data.connected_devices.forEach(d => {
      text += ` - ${d.serial} [${d.state}] ${d.info}\n`;
    });
    text += `\nActive Port Forwards:\n`;
    if (data.active_forwards.length === 0) {
      text += ` (None active yet)\n`;
    } else {
      data.active_forwards.forEach(f => {
        text += ` - ${f}\n`;
      });
    }
    box.innerText = text;
  } catch (e) {
    box.innerText = `Error: ${e.message}`;
  }
}

async function runAdbSetup() {
  const box = document.getElementById("adbDetails");
  box.innerText = "Configuring USB port forwarding for Phone 1 & Phone 2...";
  try {
    const res = await fetch("/api/adb/setup", { method: "POST" });
    const data = await res.json();
    box.innerText = JSON.stringify(data, null, 2);
    setTimeout(refreshAdbStatus, 1500);
  } catch (e) {
    box.innerText = `Setup Error: ${e.message}`;
  }
}

// Camera Management & Manual Selection
async function loadAvailableCameras() {
  try {
    const [camsRes, configRes] = await Promise.all([
      fetch("/api/cameras/available"),
      fetch("/api/config")
    ]);
    const available = await camsRes.json();
    const config = await configRes.json();
    const currentCams = config.cameras || {};

    for (let i = 1; i <= 3; i++) {
      const select = document.getElementById(`cam${i}Select`);
      if (!select) continue;

      const currentSource = String(currentCams[`cam${i}`]?.source || "");

      select.innerHTML = available.map(src => {
        const isSelected = String(src.id) === currentSource ? "selected" : "";
        return `<option value="${src.id}" ${isSelected}>${src.name}</option>`;
      }).join("");

      select.innerHTML += `<option value="custom">✏ Custom URL / Device Index...</option>`;
    }
  } catch (e) {
    console.error("Error loading camera options:", e);
  }
}

async function onCameraSelected(camId, selectedSource) {
  if (selectedSource === "custom") {
    const custom = prompt(`Enter custom device index (e.g. 0, 1) or stream URL for Cam ${camId}:`);
    if (custom) {
      selectedSource = custom.trim();
    } else {
      loadAvailableCameras();
      return;
    }
  }

  try {
    const res = await fetch("/api/cameras/assign", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cam_id: camId, source: selectedSource })
    });
    const data = await res.json();
    if (data.success) {
      const img = document.getElementById(`cam${camId}Img`);
      if (img) img.src = `/api/stream/cam/${camId}?t=${Date.now()}`;
      console.log(`Cam ${camId} updated to ${selectedSource}`);
    }
  } catch (e) {
    console.error(`Failed to assign Cam ${camId}:`, e);
  }
}

async function swapCameras(camA, camB) {
  const selA = document.getElementById(`cam${camA}Select`);
  const selB = document.getElementById(`cam${camB}Select`);
  if (!selA || !selB) return;

  const srcA = selA.value;
  const srcB = selB.value;

  selA.value = srcB;
  selB.value = srcA;

  await Promise.all([
    onCameraSelected(camA, srcB),
    onCameraSelected(camB, srcA)
  ]);
}

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
  initWebSocket();
  loadAvailableCameras();
});
