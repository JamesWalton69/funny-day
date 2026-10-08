let ws = null;
let currentLoadedVideoUrl = null;

const videoEl = document.getElementById("tvVideo");
const countdownOverlay = document.getElementById("tvCountdownOverlay");
const suspenseOverlay = document.getElementById("tvSuspenseOverlay");
const calibrateOverlay = document.getElementById("tvCalibrateOverlay");
const matchOverOverlay = document.getElementById("tvMatchOverOverlay");

const countdownNum = document.getElementById("countdownNum");
const calibTimer = document.getElementById("calibTimer");
const roundText = document.getElementById("tvRoundText");

function initTvWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log("TV screen connected to telemetry WebSocket.");
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "telemetry") {
        handleGameState(data.game_state || {});
      }
    } catch (e) {
      console.error("TV parsing error:", e);
    }
  };

  ws.onclose = () => {
    setTimeout(initTvWebSocket, 2000);
  };
}

function handleGameState(state) {
  const matchState = state.match_state || "LOBBY";
  const curRound = state.current_round || 1;
  const totRounds = state.total_rounds || 40;
  const elapsed = state.state_elapsed || 0;
  const curVid = state.current_video;

  roundText.innerText = `ROUND ${String(curRound).padStart(2, '0')} / ${totRounds}`;

  // Hide all overlays initially
  countdownOverlay.style.display = "none";
  suspenseOverlay.style.display = "none";
  calibrateOverlay.style.display = "none";
  matchOverOverlay.style.display = "none";

  if (matchState === "LOBBY") {
    countdownOverlay.style.display = "flex";
    countdownNum.innerText = "STANDBY";
    document.getElementById("countdownSub").innerText = "Waiting for referee to start match...";
    if (!videoEl.paused) {
      videoEl.pause();
    }
    videoEl.currentTime = 0;
    currentLoadedVideoUrl = null;
  }
  else if (matchState === "CALIBRATING") {
    calibrateOverlay.style.display = "flex";
    const rem = Math.max(1, Math.ceil(3.0 - elapsed));
    calibTimer.innerText = rem;
    if (!videoEl.paused) videoEl.pause();
  }
  else if (matchState === "COUNTDOWN") {
    countdownOverlay.style.display = "flex";
    const rem = Math.max(1, Math.ceil(3.0 - elapsed));
    countdownNum.innerText = rem;
    document.getElementById("countdownSub").innerText = "First challenge starting...";
    if (!videoEl.paused) videoEl.pause();
  }
  else if (matchState === "PLAYING_VIDEO") {
    if (curVid && curVid.url) {
      if (currentLoadedVideoUrl !== curVid.url) {
        currentLoadedVideoUrl = curVid.url;
        videoEl.src = curVid.url;
        videoEl.currentTime = 0;
        videoEl.play().catch(e => {
          console.warn("Autoplay interaction policy, click screen if needed:", e);
        });
      } else if (videoEl.paused) {
        videoEl.play().catch(() => {});
      }
    }
  }
  else if (matchState === "SUSPENSE_INTERVAL") {
    suspenseOverlay.style.display = "flex";
    if (!videoEl.paused) videoEl.pause();
  }
  else if (matchState === "MATCH_OVER") {
    matchOverOverlay.style.display = "flex";
    if (!videoEl.paused) videoEl.pause();
  }
}

// When video completes, signal backend
videoEl.addEventListener("ended", () => {
  console.log("Challenge video ended. Notifying referee engine...");
  fetch("/api/video_done", { method: "POST" }).catch(e => console.error(e));
});

// Enable audio / fullscreen on click
document.body.addEventListener("click", () => {
  if (videoEl.paused && videoEl.src) {
    videoEl.play().catch(() => {});
  }
});

document.addEventListener("DOMContentLoaded", () => {
  initTvWebSocket();
});
