async function loadResults() {
  try {
    const res = await fetch("/api/state");
    const data = await res.json();
    const leaderboard = data.leaderboard || [];

    // Populate Podium
    if (leaderboard.length >= 1) {
      const p1 = leaderboard[0];
      document.getElementById("pNameRank1").innerText = p1.name;
      document.getElementById("pScoreRank1").innerText = `${p1.penalty_score.toFixed(1)} pts`;
      document.getElementById("pAttRank1").innerText = `Attention: ${p1.tv_attention_pct.toFixed(1)}%`;
    }
    if (leaderboard.length >= 2) {
      const p2 = leaderboard[1];
      document.getElementById("pNameRank2").innerText = p2.name;
      document.getElementById("pScoreRank2").innerText = `${p2.penalty_score.toFixed(1)} pts`;
      document.getElementById("pAttRank2").innerText = `Attention: ${p2.tv_attention_pct.toFixed(1)}%`;
    }
    if (leaderboard.length >= 3) {
      const p3 = leaderboard[2];
      document.getElementById("pNameRank3").innerText = p3.name;
      document.getElementById("pScoreRank3").innerText = `${p3.penalty_score.toFixed(1)} pts`;
      document.getElementById("pAttRank3").innerText = `Attention: ${p3.tv_attention_pct.toFixed(1)}%`;
    }

    // Populate Table
    const tbody = document.getElementById("resultsTableBody");
    tbody.innerHTML = leaderboard.map(p => `
      <tr>
        <td><strong style="color:${getRankColor(p.rank)};">#${p.rank}</strong></td>
        <td><strong>${p.name}</strong></td>
        <td><span style="color:#10b981; font-weight:700;">${p.tv_attention_pct.toFixed(1)}%</span></td>
        <td>${p.longest_look_streak_sec.toFixed(1)}s</td>
        <td>${p.look_away_count}</td>
        <td>${p.total_away_sec.toFixed(1)}s</td>
        <td><span style="color:#f59e0b; font-weight:700;">${p.reaction_count}</span></td>
        <td><strong style="color:#f87171; font-size:1.05rem;">${p.penalty_score.toFixed(1)}</strong></td>
      </tr>
    `).join("");

  } catch (e) {
    console.error("Error loading results:", e);
  }
}

function getRankColor(rank) {
  if (rank === 1) return "#eab308";
  if (rank === 2) return "#94a3b8";
  if (rank === 3) return "#d97706";
  return "#cbd5e1";
}

document.addEventListener("DOMContentLoaded", () => {
  loadResults();
});
