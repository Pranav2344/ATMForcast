/* ============================================
   ATMForcast — Frontend Logic
   ============================================ */

let mainChart = null;

const atmSelect = document.getElementById("atmSelect");
const horizonSelect = document.getElementById("horizonSelect");
const forecastBtn = document.getElementById("forecastBtn");

const avgDemandEl = document.getElementById("avgDemand");
const totalDemandEl = document.getElementById("totalDemand");
const recommendedCashEl = document.getElementById("recommendedCash");
const tableBody = document.querySelector("#forecastTable tbody");

function formatCurrency(value) {
  return "₹" + Number(value).toLocaleString("en-IN", { maximumFractionDigits: 0 });
}

async function fetchHistory(atmId) {
  const res = await fetch(`/api/history/${atmId}`);
  if (!res.ok) throw new Error("Failed to fetch history");
  return res.json();
}

async function fetchForecast(atmId, days) {
  const res = await fetch(`/api/forecast/${atmId}?days=${days}`);
  if (!res.ok) throw new Error("Failed to fetch forecast");
  return res.json();
}

function renderChart(historyDates, historyValues, forecastDates, forecastValues) {
  const ctx = document.getElementById("mainChart").getContext("2d");

  // Bridge point: connect the last actual value to the first forecast point
  const bridgedForecastValues = [
    ...new Array(historyValues.length - 1).fill(null),
    historyValues[historyValues.length - 1],
    ...forecastValues
  ];

  const actualValues = [...historyValues, ...new Array(forecastValues.length).fill(null)];
  const allLabels = [...historyDates, ...forecastDates];

  if (mainChart) mainChart.destroy();

  mainChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: allLabels,
      datasets: [
        {
          label: "Actual",
          data: actualValues,
          borderColor: "#a8a8a8",
          backgroundColor: "rgba(168,168,168,0.08)",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.3,
          fill: true
        },
        {
          label: "Forecast",
          data: bridgedForecastValues,
          borderColor: "#ffffff",
          backgroundColor: "rgba(255,255,255,0.06)",
          borderWidth: 2,
          borderDash: [6, 4],
          pointRadius: 3,
          pointBackgroundColor: "#ffffff",
          tension: 0.3,
          fill: true
        }
      ]
    },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#1e1e1e",
          borderColor: "#2c2c2c",
          borderWidth: 1,
          titleColor: "#ffffff",
          bodyColor: "#ececec",
          callbacks: {
            label: (ctx) => ctx.raw !== null ? formatCurrency(ctx.raw) : ""
          }
        }
      },
      scales: {
        x: {
          ticks: { color: "#6b6b6b", maxRotation: 0, autoSkip: true, maxTicksLimit: 10 },
          grid: { color: "#1e1e1e" }
        },
        y: {
          ticks: {
            color: "#6b6b6b",
            callback: (val) => "₹" + (val / 1000) + "k"
          },
          grid: { color: "#1e1e1e" }
        }
      }
    }
  });
}

function renderTable(dates, values) {
  tableBody.innerHTML = "";
  dates.forEach((date, i) => {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${date}</td><td>${formatCurrency(values[i])}</td>`;
    tableBody.appendChild(row);
  });
}

function updateStats(values, recommendedCash) {
  const total = values.reduce((a, b) => a + b, 0);
  const avg = total / values.length;

  avgDemandEl.textContent = formatCurrency(avg);
  totalDemandEl.textContent = formatCurrency(total);
  recommendedCashEl.textContent = formatCurrency(recommendedCash);
}

async function runForecast() {
  const atmId = atmSelect.value;
  const days = horizonSelect.value;

  forecastBtn.disabled = true;
  forecastBtn.textContent = "Generating...";

  try {
    const [history, forecast] = await Promise.all([
      fetchHistory(atmId),
      fetchForecast(atmId, days)
    ]);

    renderChart(history.dates, history.values, forecast.dates, forecast.values);
    renderTable(forecast.dates, forecast.values);
    updateStats(forecast.values, forecast.recommended_cash_load);
    lastForecastData = forecast; 
  } catch (err) {
    console.error(err);
    alert("Something went wrong while generating the forecast. Check the console/server logs.");
  } finally {
    forecastBtn.disabled = false;
    forecastBtn.textContent = "Generate Forecast";
  }
}

forecastBtn.addEventListener("click", runForecast);

// Auto-run on page load with the first ATM in the list
window.addEventListener("DOMContentLoaded", () => {
  if (atmSelect.options.length > 0) {
    runForecast();
  }
});
async function fetchComparison(days = 7) {
  const res = await fetch(`/api/forecast-all?days=${days}`);
  if (!res.ok) throw new Error("Failed to fetch comparison data");
  return res.json();
}

function renderComparisonTable(atms) {
  const body = document.querySelector("#comparisonTable tbody");
  body.innerHTML = "";

  atms.forEach((atm, index) => {
    const rank = index + 1;
    const row = document.createElement("tr");
    row.innerHTML = `
      <td><span class="rank-badge ${rank === 1 ? 'top' : ''}">${rank}</span></td>
      <td>${atm.atm_id}</td>
      <td>${formatCurrency(atm.total_demand)}</td>
      <td>${formatCurrency(atm.recommended_cash_load)}</td>
    `;
    body.appendChild(row);
  });
}

async function loadComparison() {
  try {
    const data = await fetchComparison(7);
    renderComparisonTable(data.atms);
  } catch (err) {
    console.error(err);
  }
}

// Load comparison table once on page load, alongside the existing forecast
window.addEventListener("DOMContentLoaded", () => {
  loadComparison();
});
let lastForecastData = null; // stores the most recent forecast for export

function downloadCSV() {
  if (!lastForecastData) {
    alert("Generate a forecast first before downloading.");
    return;
  }

  const { atm_id, dates, values, recommended_cash_load } = lastForecastData;

  let csv = "ATM ID,Date,Predicted Cash Demand (INR)\n";
  dates.forEach((date, i) => {
    csv += `${atm_id},${date},${values[i]}\n`;
  });
  csv += `\nRecommended Cash Load (INR),${recommended_cash_load}\n`;

  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${atm_id}_forecast_report.csv`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

document.getElementById("downloadBtn").addEventListener("click", downloadCSV);
document.getElementById("uploadBtn").addEventListener("click", async () => {
  const fileInput = document.getElementById("csvUpload");
  const statusEl = document.getElementById("uploadStatus");

  if (!fileInput.files.length) {
    statusEl.textContent = "Please select a CSV file first.";
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  statusEl.textContent = "Uploading & retraining... this may take a moment.";

  try {
    const res = await fetch("/api/upload-dataset", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) throw new Error(data.error || "Upload failed");

    statusEl.textContent = `✅ Dataset updated (${data.total_rows} rows). Model retrained.`;
    location.reload(); // refresh ATM dropdown + forecast with new data
  } catch (err) {
    statusEl.textContent = `❌ ${err.message}`;
  }
});